import argparse
import joblib
import torch
from PIL import Image
from torchvision import models, transforms
from .config import CLASSIFIER_MODEL_DIR, IMAGE_SIZE

def main():
    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("--model", default="gan_augmented_random_forest")
    args = p.parse_args()
    bundle = joblib.load(CLASSIFIER_MODEL_DIR/f"{args.model}.joblib")
    model, classes = bundle["model"], bundle["classes"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        net = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    except Exception:
        net = models.mobilenet_v2(weights=None)
    net.features.to(device).eval()
    tf = transforms.Compose([transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)), transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    with torch.no_grad():
        x = tf(Image.open(args.image).convert("RGB")).unsqueeze(0).to(device)
        feat = net.features(x).mean((2,3)).cpu().numpy()
    pred = model.predict(feat)[0]
    print("Predicted class:", classes[int(pred)])

if __name__ == "__main__":
    main()
