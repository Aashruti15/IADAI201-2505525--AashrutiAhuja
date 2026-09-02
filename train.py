"""
train.py
--------
Trains a YOLOv8 model on your parking-slot dataset (Roboflow export).

HOW TO USE:
1. Put this file in the SAME folder as your extracted dataset, so the
   folder looks like:

   ParkVisionAI/
   ├── train.py
   ├── data.yaml        <-- comes from your Roboflow zip
   ├── train/
   ├── valid/
   └── test/

2. Install requirements first (see requirements.txt / README).
3. Run:  python train.py
4. When it finishes, your trained model will be saved at:
   runs/detect/train/weights/best.pt
   Copy that file into this same folder and rename it to best.pt
   (the app.py file expects it there).

NOTE: Training is much faster with a GPU. If you don't have one,
run this same code in Google Colab (free GPU) instead of your laptop,
then download the resulting best.pt file back to this folder.
"""

from ultralytics import YOLO
import yaml
import os
import random
import shutil
from pathlib import Path

DATA_YAML = "yolo_dataset/data.yaml"

# --- Speed settings for laptops WITHOUT a dedicated GPU ---
# Your assignment only requires 100+ images per class, so we don't need
# to train on all 8,691 images. This subsamples the dataset down to a
# small, fast-to-train size.
IMAGES_PER_SPLIT = {"train": 400, "valid": 100, "test": 60}
EPOCHS = 20
IMG_SIZE = 416   # smaller than 640 = noticeably faster on CPU
BATCH_SIZE = 8


def make_small_dataset():
    """Creates yolo_dataset_small/ with a random subset of images+labels
    copied from yolo_dataset/, so training has far less data to chew through."""
    src_root = Path("yolo_dataset")
    dst_root = Path("yolo_dataset_small")

    if dst_root.exists():
        print(f"{dst_root} already exists, reusing it (delete the folder to regenerate).")
        return dst_root / "data.yaml"

    for split, limit in IMAGES_PER_SPLIT.items():
        src_img_dir = src_root / split / "images"
        src_lbl_dir = src_root / split / "labels"
        dst_img_dir = dst_root / split / "images"
        dst_lbl_dir = dst_root / split / "labels"
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        all_images = list(src_img_dir.glob("*.*"))
        random.shuffle(all_images)
        chosen = all_images[:limit]

        for img_path in chosen:
            shutil.copy(img_path, dst_img_dir / img_path.name)
            lbl_path = src_lbl_dir / (img_path.stem + ".txt")
            if lbl_path.exists():
                shutil.copy(lbl_path, dst_lbl_dir / lbl_path.name)

        print(f"{split}: copied {len(chosen)} images into {dst_img_dir}")

    # Copy data.yaml and fix it to point at the new folder
    with open(src_root / "data.yaml", "r") as f:
        content = f.read()
    with open(dst_root / "data.yaml", "w") as f:
        f.write(content)

    return dst_root / "data.yaml"


def main():
    if not os.path.exists(DATA_YAML):
        raise FileNotFoundError(
            f"Could not find {DATA_YAML} in this folder. "
            "Make sure train.py sits next to your extracted dataset's data.yaml file."
        )

    small_yaml = make_small_dataset()

    # Show the classes so you can confirm they look right (e.g. space-empty, space-occupied)
    with open(small_yaml, "r") as f:
        data_cfg = yaml.safe_load(f)
    print("Classes found in dataset:", data_cfg.get("names"))

    # Load a small pretrained YOLOv8 model (fast, good for a class project)
    model = YOLO("yolov8n.pt")

    # Train
    model.train(
        data=str(small_yaml),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        patience=10,      # stops early if it stops improving
        name="parkvision_train"
    )

    # Evaluate on the test/val set
    metrics = model.val()
    print("Validation metrics:", metrics)

    print("\nDONE. Your trained weights are at:")
    print("runs/detect/parkvision_train/weights/best.pt")
    print("Copy that file into this project folder and rename it to best.pt")

if __name__ == "__main__":
    main()