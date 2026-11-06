import argparse
import torch
from torchvision.utils import save_image
from .config import GAN_MODEL_DIR, GAN_DATA_DIR, LATENT_DIM
from .data_utils import get_classes
from .config import SPLIT_DIR
from .gan_models import Generator

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per-class", type=int, default=200)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classes = get_classes(SPLIT_DIR/"train")
    for cls in classes:
        model_path = GAN_MODEL_DIR/cls/"generator.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing {model_path}. Train GAN first.")
        G = Generator(LATENT_DIM).to(device)
        G.load_state_dict(torch.load(model_path, map_location=device))
        G.eval()
        out = GAN_DATA_DIR/cls
        out.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            for start in range(0, args.per_class, 32):
                n = min(32, args.per_class-start)
                imgs = (G(torch.randn(n, LATENT_DIM, 1, 1, device=device))+1)/2
                for j, img in enumerate(imgs):
                    save_image(img, out/f"synthetic_{start+j:05d}.jpg")
        print(f"Generated {args.per_class} images for {cls} in {out}")

if __name__ == "__main__":
    main()
