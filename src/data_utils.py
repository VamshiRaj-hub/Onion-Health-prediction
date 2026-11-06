from pathlib import Path
from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from .config import IMAGE_SIZE, SEED

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def get_classes(split_dir):
    split_dir = Path(split_dir)
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing dataset folder: {split_dir}")
    classes = sorted(p.name for p in split_dir.iterdir() if p.is_dir())
    if len(classes) < 2:
        raise ValueError(
            f"Found {len(classes)} class folder(s) in {split_dir}. "
            "A classification project needs at least two labeled condition classes."
        )
    return classes

def image_files(folder):
    folder = Path(folder)
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)

class OnionDataset(Dataset):
    def __init__(self, split_dir, classes, augment=False):
        self.items = []
        self.classes = classes
        self.class_to_idx = {c:i for i,c in enumerate(classes)}
        for cls in classes:
            for p in image_files(Path(split_dir) / cls):
                self.items.append((p, self.class_to_idx[cls]))
        if not self.items:
            raise ValueError(f"No image files found under {split_dir}")
        ops = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
        if augment:
            ops += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(10)]
        ops += [transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
        self.transform = transforms.Compose(ops)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        try:
            img = Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            raise RuntimeError(f"Cannot read image: {path}") from e
        return self.transform(img), label

def seed_everything(seed=SEED):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
