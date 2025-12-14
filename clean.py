import os
from PIL import Image, UnidentifiedImageError

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")

def is_image(fn: str) -> bool:
    return fn.lower().endswith(IMAGE_EXTS)

def clean_folder(root: str, delete: bool = True):
    bad = []
    total = 0

    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if not is_image(fn):
                continue
            path = os.path.join(dirpath, fn)
            total += 1
            try:
                # verify() checks file integrity without decoding full image
                with Image.open(path) as img:
                    img.verify()
            except (UnidentifiedImageError, OSError, ValueError) as e:
                bad.append((path, str(e)))
                if delete:
                    try:
                        os.remove(path)
                    except Exception:
                        pass

    print(f"[CLEAN] Scanned {total} images under: {root}")
    print(f"[CLEAN] Bad images found: {len(bad)}")
    if bad:
        print("[CLEAN] Examples:")
        for p, e in bad[:10]:
            print("  -", p, "|", e)

if __name__ == "__main__":
    # Clean both raw + split (raw is the important one)
    clean_folder("medicine_boxes_raw", delete=True)
    clean_folder("medicine_boxes_split", delete=True)
