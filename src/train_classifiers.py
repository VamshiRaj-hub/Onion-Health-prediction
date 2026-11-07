import json
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torchvision import models, transforms
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
import joblib
import matplotlib.pyplot as plt
from .config import SPLIT_DIR, GAN_DATA_DIR, CLASSIFIER_MODEL_DIR, REPORT_DIR, IMAGE_SIZE
from .data_utils import get_classes, image_files, seed_everything

TF = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def load_backbone(device):
    try:
        weights = models.MobileNet_V2_Weights.DEFAULT
        model = models.mobilenet_v2(weights=weights)
    except Exception as e:
        print("Could not download pretrained MobileNetV2 weights; using untrained backbone:", e)
        model = models.mobilenet_v2(weights=None)
    feature_model = model.features.to(device).eval()
    return feature_model

def features_for(paths, feature_model, device):
    feats, good = [], []
    with torch.no_grad():
        for path in paths:
            try:
                x = TF(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
                f = feature_model(x).mean(dim=(2,3)).cpu().numpy().ravel()
                feats.append(f); good.append(path)
            except Exception as e:
                print("Skipping unreadable image", path, e)
    return np.asarray(feats), good

def collect_split(split, classes, feature_model, device, include_gan=False):
    paths, labels = [], []
    for i, cls in enumerate(classes):
        real = image_files(SPLIT_DIR/split/cls)
        paths.extend(real); labels.extend([i]*len(real))
        if include_gan and split == "train":
            synth = image_files(GAN_DATA_DIR/cls)
            paths.extend(synth); labels.extend([i]*len(synth))
    X, good = features_for(paths, feature_model, device)
    # Keep labels aligned with successfully read files.
    good_set = set(good)
    y = np.asarray([lab for path, lab in zip(paths, labels) if path in good_set])
    return X, y

def evaluate(name, model, X_train, y_train, X_val, y_val, X_test, y_test, classes):
    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    result = {
        "validation_accuracy": float(accuracy_score(y_val, val_pred)),
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "test_report": classification_report(y_test, test_pred, labels=list(range(len(classes))),
                    target_names=classes, output_dict=True, zero_division=0)
    }
    print(f"\n{name} | validation accuracy: {result['validation_accuracy']:.4f} | test accuracy: {result['test_accuracy']:.4f}")
    print(classification_report(y_test, test_pred, labels=list(range(len(classes))),
                                target_names=classes, zero_division=0))
    cm = confusion_matrix(y_test, test_pred, labels=list(range(len(classes))))
    fig, ax = plt.subplots(figsize=(6,5))
    ax.imshow(cm)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(name)
    fig.tight_layout()
    fig.savefig(REPORT_DIR/f"confusion_matrix_{name}.png")
    plt.close(fig)
    joblib.dump({"model": model, "classes": classes}, CLASSIFIER_MODEL_DIR/f"{name}.joblib")
    return result

def main():
    seed_everything()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classes = get_classes(SPLIT_DIR/"train")
    for split in ["val", "test"]:
        found = get_classes(SPLIT_DIR/split)
        if found != classes:
            raise ValueError(f"Class mismatch in {split}: {found}; expected {classes}")
    fm = load_backbone(device)
    print("Extracting real training features...")
    Xtr, ytr = collect_split("train", classes, fm, device, False)
    print("Extracting GAN-augmented training features...")
    Xaug, yaug = collect_split("train", classes, fm, device, True)
    print("Extracting validation and test features...")
    Xv, yv = collect_split("val", classes, fm, device)
    Xt, yt = collect_split("test", classes, fm, device)
    if min(len(Xtr),len(Xv),len(Xt)) == 0:
        raise ValueError("Train, validation, and test must each contain readable images.")
    results = {}
    models_to_try = {
        "random_forest": lambda: RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"),
        "svm": lambda: make_pipeline(StandardScaler(), SVC(kernel="rbf", C=3, class_weight="balanced"))
    }
    try:
        from xgboost import XGBClassifier
        models_to_try["xgboost"] = lambda: XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, objective="multi:softmax",
            num_class=len(classes), eval_metric="mlogloss", random_state=42)
    except ImportError:
        print("XGBoost not installed; RF and SVM will still run. Optional: pip install xgboost")
    for dataset_name, (a,b) in {"baseline":(Xtr,ytr), "gan_augmented":(Xaug,yaug)}.items():
        results[dataset_name] = {}
        for model_name, factory in models_to_try.items():
            results[dataset_name][model_name] = evaluate(
                f"{dataset_name}_{model_name}", factory(), a,b,Xv,yv,Xt,yt,classes)
    with open(REPORT_DIR/"metrics.json","w") as f:
        json.dump({"classes":classes,"results":results}, f, indent=2)
    print(f"\nSaved metrics to {REPORT_DIR/'metrics.json'}")

if __name__ == "__main__":
    main()
