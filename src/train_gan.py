import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from .config import SPLIT_DIR, GAN_MODEL_DIR, LATENT_DIM, IMAGE_SIZE, NUM_WORKERS
from .data_utils import OnionDataset, get_classes, seed_everything
from .gan_models import Generator, Discriminator

def train_class(cls, epochs, batch_size, device):
    ds = OnionDataset(SPLIT_DIR / "train" / cls, ["_"], augment=False) if False else None
    # Use only images inside this class folder; assign a dummy label.
    class SingleClass(torch.utils.data.Dataset):
        def __init__(self, folder):
            from PIL import Image
            from .data_utils import image_files
            from torchvision import transforms
            self.paths = image_files(folder)
            self.tf = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,)*3, (0.5,)*3)
            ])
        def __len__(self): return len(self.paths)
        def __getitem__(self, i):
            from PIL import Image
            return self.tf(Image.open(self.paths[i]).convert("RGB"))
    dataset = SingleClass(SPLIT_DIR / "train" / cls)
    if len(dataset) < 10:
        raise ValueError(f"Class '{cls}' has only {len(dataset)} training images; need more for GAN training.")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        drop_last=True, num_workers=NUM_WORKERS)
    G, D = Generator(LATENT_DIM).to(device), Discriminator().to(device)
    criterion = nn.BCELoss()
    opt_g = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
    out_dir = GAN_MODEL_DIR / cls
    out_dir.mkdir(parents=True, exist_ok=True)
    fixed = torch.randn(16, LATENT_DIM, 1, 1, device=device)
    for epoch in range(1, epochs+1):
        for real in loader:
            real = real.to(device)
            b = real.size(0)
            ones = torch.ones(b, device=device)
            zeros = torch.zeros(b, device=device)
            opt_d.zero_grad()
            real_loss = criterion(D(real), ones)
            noise = torch.randn(b, LATENT_DIM, 1, 1, device=device)
            fake = G(noise)
            fake_loss = criterion(D(fake.detach()), zeros)
            d_loss = real_loss + fake_loss
            d_loss.backward()
            opt_d.step()
            opt_g.zero_grad()
            noise = torch.randn(b, LATENT_DIM, 1, 1, device=device)
            generated = G(noise)
            g_loss = criterion(D(generated), ones)
            g_loss.backward()
            opt_g.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"{cls} | epoch {epoch}/{epochs} | D={d_loss.item():.4f} G={g_loss.item():.4f}")
            with torch.no_grad():
                save_image((G(fixed).cpu()+1)/2, out_dir/f"preview_{epoch:04d}.png", nrow=4)
    torch.save(G.state_dict(), out_dir/"generator.pt")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    seed_everything()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Training device:", device)
    classes = get_classes(SPLIT_DIR/"train")
    for cls in classes:
        print(f"\nTraining GAN for class: {cls}")
        train_class(cls, args.epochs, args.batch_size, device)

if __name__ == "__main__":
    main()
