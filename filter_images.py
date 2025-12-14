import os
import cv2
import numpy as np

def colorfulness(img):
    # Hasler-Susstrunk colorfulness metric
    B, G, R = cv2.split(img.astype("float"))
    rg = np.abs(R - G)
    yb = np.abs(0.5*(R + G) - B)
    std_rg, std_yb = np.std(rg), np.std(yb)
    mean_rg, mean_yb = np.mean(rg), np.mean(yb)
    return np.sqrt(std_rg**2 + std_yb**2) + 0.3*np.sqrt(mean_rg**2 + mean_yb**2)

def foreground_fraction(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Invert threshold: foreground tends to be darker than white background
    _, th = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
    return (th > 0).mean()

def looks_like_line_diagram(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    edge_frac = (edges > 0).mean()
    # diagrams often have a lot of edges but low color
    return edge_frac > 0.03

def filter_folder(folder, dry_run=True):
    removed = 0
    kept = 0

    for fn in os.listdir(folder):
        if not fn.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            continue
        path = os.path.join(folder, fn)
        img = cv2.imread(path)
        if img is None:
            continue

        h, w = img.shape[:2]
        if min(h, w) < 180:  # too small = likely not useful
            remove = True
        else:
            c = colorfulness(img)
            fg = foreground_fraction(img)
            diag = looks_like_line_diagram(img)

            # Tuneable rules:
            # - Packaging photos tend to have fg area and some color
            # - Diagrams have low color + high edge fraction
            remove = (c < 8 and diag) or (fg < 0.05)

        if remove:
            removed += 1
            if not dry_run:
                os.remove(path)
        else:
            kept += 1

    print(f"[{folder}] kept={kept}, removed={removed}, dry_run={dry_run}")

if __name__ == "__main__":
    # Example: run on all class folders under medicine_boxes_raw/
    root = "medicine_boxes_raw"
    for cls in os.listdir(root):
        folder = os.path.join(root, cls)
        if os.path.isdir(folder):
            filter_folder(folder, dry_run=False)   # first pass: see counts
    # Then rerun with dry_run=False once you like the behavior
