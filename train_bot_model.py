"""
Train a custom YOLO model on auto-labeled combat bot data.
============================================================

The training data is collected automatically while you process videos
in CSRT/manual tracking mode. Each time the tracker is confident about
a bot's position, it saves the frame + bounding box annotation.

Usage:
    1. Process several fight videos with the GUI (manual/CSRT mode).
       Training data accumulates in ./training_data/
    2. Split ~10-15% of images into a validation set (this script does it
       automatically if no val set exists).
    3. Run this script:
           python train_bot_model.py
    4. Use the trained model:
           python gui.py   (then change the model or set it in code)
       Or:
           python smart_cropper.py -i fight.mp4 -m runs/detect/bot_tracker/weights/best.pt
"""

import os
import random
import shutil
from pathlib import Path


def prepare_val_split(data_dir: str, val_ratio: float = 0.15):
    """Split training data into train/val if no val set exists."""
    train_img_dir = Path(data_dir) / "images" / "train"
    val_img_dir = Path(data_dir) / "images" / "val"
    train_lbl_dir = Path(data_dir) / "labels" / "train"
    val_lbl_dir = Path(data_dir) / "labels" / "val"

    # Skip if val already has images
    if val_img_dir.exists() and any(val_img_dir.glob("*.jpg")):
        n_train = len(list(train_img_dir.glob("*.jpg")))
        n_val = len(list(val_img_dir.glob("*.jpg")))
        print(f"[INFO] Dataset already split: {n_train} train / {n_val} val")
        return

    val_img_dir.mkdir(parents=True, exist_ok=True)
    val_lbl_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(train_img_dir.glob("*.jpg"))
    if not images:
        print("[ERROR] No training images found in training_data/images/train/")
        print("        Process some videos in CSRT mode first to collect data.")
        raise SystemExit(1)

    # Randomly select images for validation
    n_val = max(1, int(len(images) * val_ratio))
    val_images = random.sample(images, n_val)

    for img_path in val_images:
        lbl_path = train_lbl_dir / img_path.with_suffix(".txt").name

        shutil.move(str(img_path), str(val_img_dir / img_path.name))
        if lbl_path.exists():
            shutil.move(str(lbl_path), str(val_lbl_dir / lbl_path.name))

    n_train = len(images) - n_val
    print(f"[INFO] Split dataset: {n_train} train / {n_val} val")


def train():
    from ultralytics import YOLO

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "training_data")
    data_yaml = os.path.join(data_dir, "data.yaml")

    if not os.path.exists(data_yaml):
        print("[ERROR] training_data/data.yaml not found.")
        print("        Process some videos first to generate training data.")
        raise SystemExit(1)

    # Auto-split if needed
    prepare_val_split(data_dir)

    # Count images
    train_count = len(list(Path(data_dir, "images", "train").glob("*.jpg")))
    val_count = len(list(Path(data_dir, "images", "val").glob("*.jpg")))
    print(f"\n[INFO] Training dataset: {train_count} train + {val_count} val images")
    print(f"[INFO] Class: bot (index 0)\n")

    # Load base model and fine-tune
    base_model = os.path.join(script_dir, "yolo26n.pt")
    if not os.path.exists(base_model):
        print(f"[WARN] Base model {base_model} not found, using default yolo11n.pt")
        base_model = "yolo11n.pt"

    model = YOLO(base_model)

    # Determine epochs based on dataset size
    if train_count < 100:
        epochs = 100
        print(f"[INFO] Small dataset ({train_count} images) — using {epochs} epochs")
        print("[TIP]  Process more videos to improve accuracy!\n")
    elif train_count < 500:
        epochs = 50
    else:
        epochs = 30

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,
        batch=-1,         # auto batch size based on GPU memory
        project=os.path.join(script_dir, "runs", "detect"),
        name="bot_tracker",
        exist_ok=True,    # overwrite previous runs
        patience=10,      # early stopping
        save=True,
        plots=True,
        verbose=True,
    )

    best_path = Path(script_dir) / "runs" / "detect" / "bot_tracker" / "weights" / "best.pt"
    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Best model: {best_path}")
    print(f"  ")
    print(f"  To use it:")
    print(f"    python smart_cropper.py -i fight.mp4 -m \"{best_path}\"")
    print(f"  ")
    print(f"  Or copy it to your project directory:")
    print(f"    copy \"{best_path}\" bot_tracker.pt")
    print(f"    python smart_cropper.py -i fight.mp4 -m bot_tracker.pt")
    print(f"{'='*60}")


if __name__ == "__main__":
    train()
