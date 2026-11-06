import streamlit as st
import joblib, torch
from PIL import Image
from torchvision import models, transforms
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IMAGE_SIZE = 128
st.set_page_config(page_title="Onion Quality Checker", page_icon="🧅")
st.title("🧅 Onion Quality Classification")
model_files = sorted((ROOT/"models"/"classifiers").glob("*.joblib"))
if not model_files:
    st.warning("Train the classifiers first: python -m src.train_classifiers")
    st.stop()
chosen = st.selectbox("Choose classifier", model_files, format_func=lambda p:p.stem)
uploaded = st.file_uploader("Upload an onion image", type=["jpg","jpeg","png","bmp","webp"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded onion", width=300)
    bundle = joblib.load(chosen)
    clf, classes = bundle["model"], bundle["classes"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    except Exception:
        backbone = models.mobilenet_v2(weights=None)
    backbone.features.to(device).eval()
    tf = transforms.Compose([transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)), transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    with torch.no_grad():
        x = tf(img).unsqueeze(0).to(device)
        feat = backbone.features(x).mean((2,3)).cpu().numpy()
    pred = int(clf.predict(feat)[0])
    st.success(f"Predicted class: {classes[pred]}")
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(feat)[0]
        st.subheader("Class scores")
        for c, score in zip(classes, probs):
            st.write(f"{c}: {score:.1%}")
