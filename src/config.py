from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SPLIT_DIR = DATA_DIR / "split"
PROCESSED_DIR = DATA_DIR / "processed"
GAN_DATA_DIR = PROCESSED_DIR / "gan_augmented"
MODEL_DIR = ROOT / "models"
GAN_MODEL_DIR = MODEL_DIR / "gan"
CLASSIFIER_MODEL_DIR = MODEL_DIR / "classifiers"
REPORT_DIR = ROOT / "reports"

IMAGE_SIZE = 128
LATENT_DIM = 100
NUM_WORKERS = 0  # Windows-friendly
SEED = 42

for path in [PROCESSED_DIR, GAN_DATA_DIR, MODEL_DIR, GAN_MODEL_DIR,
             CLASSIFIER_MODEL_DIR, REPORT_DIR]:
    path.mkdir(parents=True, exist_ok=True)
