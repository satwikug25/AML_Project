import os
import random
import csv
from shutil import copy2

# =========================
# CONFIG
# =========================

RAW_ROOT = "medicine_boxes_raw"
SPLIT_ROOT = "medicine_boxes_split"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15  # test = remainder

RANDOM_SEED = 42
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


# =========================
# HELPERS
# =========================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def is_image(fname: str) -> bool:
    return fname.lower().endswith(IMAGE_EXTS)


# =========================
# SPLIT LOGIC
# =========================

def split_dataset():
    print("📂 Creating train/val/test split from RAW data...")
    random.seed(RANDOM_SEED)

    if not os.path.isdir(RAW_ROOT):
        raise RuntimeError(f"RAW_ROOT not found: {RAW_ROOT}")

    # Prepare split folders
    for split in ["train", "val", "test"]:
        ensure_dir(os.path.join(SPLIT_ROOT, split))

    index_rows = [("relative_path", "class", "split")]

    classes = sorted(
        d for d in os.listdir(RAW_ROOT)
        if os.path.isdir(os.path.join(RAW_ROOT, d))
    )

    if not classes:
        raise RuntimeError("No class folders found in RAW_ROOT")

    print(f"Found {len(classes)} classes")

    for cls in classes:
        raw_cls_dir = os.path.join(RAW_ROOT, cls)
        images = [f for f in os.listdir(raw_cls_dir) if is_image(f)]

        if len(images) < 3:
            print(f"⚠️  Skipping '{cls}' (only {len(images)} images)")
            continue

        random.shuffle(images)
        n = len(images)

        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        n_test = n - n_train - n_val

        # Safety: ensure at least 1 train image
        if n_train == 0:
            n_train = 1
            if n_val > 0:
                n_val -= 1
            else:
                n_test = max(n_test - 1, 0)

        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]

        print(
            f"Class '{cls}': total={n}, "
            f"train={len(train_imgs)}, val={len(val_imgs)}, test={len(test_imgs)}"
        )

        for split_name, split_imgs in [
            ("train", train_imgs),
            ("val", val_imgs),
            ("test", test_imgs),
        ]:
            split_cls_dir = os.path.join(SPLIT_ROOT, split_name, cls)
            ensure_dir(split_cls_dir)

            for img in split_imgs:
                src = os.path.join(raw_cls_dir, img)
                dst = os.path.join(split_cls_dir, img)
                copy2(src, dst)

                rel_path = os.path.join(split_name, cls, img)
                index_rows.append((rel_path, cls, split_name))

    # Write index CSV
    csv_path = os.path.join(SPLIT_ROOT, "dataset_index.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(index_rows)

    print("\n✅ Split complete!")
    print(f"Dataset folder: {SPLIT_ROOT}/")
    print(f"Index file: {csv_path}")


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    split_dataset()
