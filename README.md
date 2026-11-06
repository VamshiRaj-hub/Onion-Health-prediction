# Onion quality classification with GAN augmentation

This project trains a separate unconditional DCGAN for each class using that class's
**training images only**. It generates extra training images, extracts MobileNetV2 features,
and compares Random Forest, SVM, and (optionally) XGBoost classifiers. Validation and test
images remain real images.

## 1. Dataset layout

Put your labeled images in one folder per class. For example:

```
data/raw/
  healthy/
    image1.jpg
    image2.jpg
  unhealthy/
    image1.jpg
    image2.jpg
```

The folder names are the class labels; use names that accurately describe the images. The code
splits these class folders into `data/split/train`, `data/split/val`, and `data/split/test`:

```powershell
python -m src.split_dataset data/raw
```

By default, the split is reproducible (seed 42) and approximately 70% train, 15% validation,
and 15% test, with at least one image per class in each split. Use `--seed` to choose another
reproducible shuffle. The command copies images and does not overwrite a non-empty `data/split`;
move or clear an old split before running it again. Keep photos of the same physical onion in
only one split to avoid data leakage.

## 2. Windows setup (VS Code terminal)

```powershell
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `torch` installation fails, install the matching PyTorch build from https://pytorch.org/get-started/locally/

## 3. Run in order

```powershell
python -m src.split_dataset data/raw
python -m src.check_dataset
python -m src.train_gan --epochs 100 --batch-size 32
python -m src.generate_images --per-class 200
python -m src.train_classifiers
streamlit run app.py
```

For a first test, use `--epochs 5` and `--per-class 10` to verify the pipeline.
GAN training is computationally expensive; use a GPU if available. Increase epochs only after
checking generated samples. Generated images are saved in `data/processed/gan_augmented/`.

## 4. Resolution note

If the downloaded images are 32x32, resizing them to 128x128 does not restore lost detail.
Prefer downloading at 128x128 or higher if Images.cv offers it. The code resizes to 128x128
for consistent model input.

## 5. Outputs

- `models/gan/<class>/generator.pt`: trained generator per class
- `data/processed/gan_augmented/<class>/`: generated training images
- `models/classifiers/`: trained classifier pipelines
- `reports/metrics.json`: validation and test metrics
- `reports/confusion_matrix_*.png`: confusion matrices

## Methodology cautions

- Train GANs only on the training split.
- Evaluate only on the untouched real test split.
- Check generated samples visually; GANs can produce unrealistic or duplicated images.
- Keep images of the same physical onion in only one split to avoid data leakage.
- Compare baseline (real train only) and augmented (real + synthetic train) results.
- Report per-class precision, recall, F1-score and confusion matrix, not accuracy alone.
