import argparse
import random
import shutil
from pathlib import Path

from .config import SPLIT_DIR
from .data_utils import image_files


def _is_within(path, parent):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def split_dataset(source_dir, output_dir, seed):
    source_dir = Path(source_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source folder does not exist: {source_dir}")
    if _is_within(output_dir, source_dir) or _is_within(source_dir, output_dir):
        raise ValueError("Source and output folders must not contain one another.")

    classes = sorted(path for path in source_dir.iterdir() if path.is_dir())
    if len(classes) < 2:
        raise ValueError(
            f"Expected at least two class folders under {source_dir}; found {len(classes)}."
        )

    files_by_class = {}
    for class_dir in classes:
        files = image_files(class_dir)
        if len(files) < 3:
            raise ValueError(
                f"Class '{class_dir.name}' has {len(files)} images; "
                "at least 3 are needed to create train, val, and test splits."
            )
        files_by_class[class_dir] = files

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output folder is not empty: {output_dir}. Move its contents elsewhere "
            "before creating a new split."
        )

    rng = random.Random(seed)
    for class_dir, files in files_by_class.items():
        rng.shuffle(files)
        train_count = max(1, int(len(files) * 0.70))
        val_count = max(1, int(len(files) * 0.15))
        if train_count + val_count >= len(files):
            train_count = len(files) - 2
        partitions = {
            "train": files[:train_count],
            "val": files[train_count:train_count + val_count],
            "test": files[train_count + val_count:],
        }
        for split, split_files in partitions.items():
            for source_file in split_files:
                relative_path = source_file.relative_to(class_dir)
                destination = output_dir / split / class_dir.name / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_file, destination)
        print(
            f"{class_dir.name}: "
            + ", ".join(f"{split}={len(items)}" for split, items in partitions.items())
        )

    print(f"Dataset splits created at: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Split labeled class folders into train, val, and test datasets."
    )
    parser.add_argument(
        "source",
        type=Path,
        help="Folder containing one subfolder per class, e.g. data/raw/",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    split_dataset(args.source, SPLIT_DIR, args.seed)


if __name__ == "__main__":
    main()
