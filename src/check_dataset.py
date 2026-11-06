from collections import Counter
from PIL import Image
from .config import SPLIT_DIR
from .data_utils import get_classes, image_files

def main():
    if not SPLIT_DIR.exists():
        raise FileNotFoundError(
            f"Dataset not found at {SPLIT_DIR}. Copy the extracted Images.cv `data` "
            "folder here and rename it `split`."
        )
    split_classes = {}
    for split in ["train", "val", "test"]:
        folder = SPLIT_DIR / split
        classes = get_classes(folder)
        split_classes[split] = classes
        counts = {}
        for cls in classes:
            imgs = image_files(folder / cls)
            counts[cls] = len(imgs)
        print(f"\n{split.upper()} ({sum(counts.values())} images)")
        for cls, n in counts.items():
            print(f"  {cls}: {n}")
    base = split_classes["train"]
    for split, classes in split_classes.items():
        if classes != base:
            raise ValueError(f"Class folders in {split} do not match train: {classes} vs {base}")
    first = next((p for c in base for p in image_files(SPLIT_DIR/'train'/c)), None)
    if first:
        with Image.open(first) as im:
            print(f"\nExample image: {first.name}, size={im.size}, mode={im.mode}")
    print("\nDataset structure looks valid.")

if __name__ == "__main__":
    main()
