import os
import random
import csv
import requests
from shutil import copy2



MEDICINES = [
    "ibuprofen",
    "acetaminophen",
    "naproxen",
    "cetirizine",
    "loratadine",
    "amoxicillin",
    "azithromycin",
    "omeprazole",
    "lisinopril",
    "metformin",
    "atorvastatin",
    "simvastatin",
    "aspirin",
    "clopidogrel",
    "amlodipine",
    "losartan",
    "pantoprazole",
    "furosemide",
    "levothyroxine",
    "albuterol",
]

RAW_ROOT = "medicine_boxes_raw"

SPLIT_ROOT = "medicine_boxes_split"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15  

PAGES_PER_DRUG = 8          
PAGESIZE = 50               
MAX_IMAGES_PER_DRUG = 450   

RANDOM_SEED = 42

BAD_KEYWORDS = [
    "chemical", "structure", "diagram", "line", "logo", "formula",
    "molecular", "bond", "structural", "reaction", "schematic",
    "package insert", "labeling", "leaflet", "carton", "barcode"
]



SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "course-project/1.0 (medicine dataset builder)"
})


def get_json(url, params=None):
    r = SESSION.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()



def should_skip_media_item(m, url: str) -> bool:
   
    desc = (m.get("description") or "").lower()
    title = (m.get("title") or "").lower()
    url_l = (url or "").lower()

    if any(k in desc or k in title or k in url_l for k in BAD_KEYWORDS):
        return True

    # Also skip suspicious filetypes in URL (sometimes svg/pdf-like)
    if any(ext in url_l for ext in [".svg", ".pdf"]):
        return True

    return False


def scrape_medicine_images(drug_name: str,
                           pages: int = PAGES_PER_DRUG,
                           pagesize: int = PAGESIZE,
                           max_images_per_drug: int = MAX_IMAGES_PER_DRUG) -> int:
    print(f"\n🔍 Collecting images for: {drug_name}")

    class_folder = os.path.join(RAW_ROOT, drug_name.replace(" ", "_"))
    os.makedirs(class_folder, exist_ok=True)

    spls_url = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json"

    downloaded = 0
    seen_setids = set()

    for page in range(1, pages + 1):
        if downloaded >= max_images_per_drug:
            break

        params = {
            "drug_name": drug_name,
            "pagesize": pagesize,
            "page": page,
        }

        try:
            data = get_json(spls_url, params=params)
        except Exception as e:
            print(f"  [WARN] Failed SPL list for {drug_name} page {page}: {e}")
            continue

        spls = data.get("data", [])
        if not spls:
            print(f"  No SPLs on page {page} for {drug_name}")
            break

        for spl in spls:
            if downloaded >= max_images_per_drug:
                break

            setid = spl.get("setid")
            if not setid or setid in seen_setids:
                continue
            seen_setids.add(setid)

            media_url = f"https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{setid}/media.json"
            try:
                media_data = get_json(media_url)
            except Exception as e:
                print(f"  [WARN] Failed media for setid={setid}: {e}")
                continue

            media_items = media_data.get("data", {}).get("media", [])
            if not media_items:
                continue

            for m in media_items:
                if downloaded >= max_images_per_drug:
                    break

                mime = (m.get("mime_type") or "").lower()
                url = m.get("url")

                # only images
                if not url or not mime.startswith("image/"):
                    continue

                if should_skip_media_item(m, url):
                    continue

                
                try:
                    img_resp = SESSION.get(url, timeout=20)
                    img_resp.raise_for_status()

                    ext = mime.split("/")[-1]
                    ext = "jpg" if ext in ["jpeg", "pjpeg"] else ext

                    fname = f"{drug_name.replace(' ', '_')}_{setid}_{downloaded:05d}.{ext}"
                    out_path = os.path.join(class_folder, fname)

                    with open(out_path, "wb") as f:
                        f.write(img_resp.content)

                    downloaded += 1
                    if downloaded % 25 == 0:
                        print(f"  Downloaded {downloaded} images for {drug_name}...")

                except Exception as e:
                    print(f"    [WARN] download failed: {e}")
                    continue

    print(f"✅ Done: {drug_name} → {downloaded} images in {class_folder}")
    return downloaded




def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def split_dataset():
    print("\n📂 Creating train/val/test split...")

    random.seed(RANDOM_SEED)

    splits = ["train", "val", "test"]
    for s in splits:
        ensure_dir(os.path.join(SPLIT_ROOT, s))

    index_rows = [("filename", "class", "split")]

    classes = [d for d in os.listdir(RAW_ROOT) if os.path.isdir(os.path.join(RAW_ROOT, d))]
    if not classes:
        raise RuntimeError(f"No class folders found in {RAW_ROOT}. Did scraping run?")

    for cls in sorted(classes):
        class_raw_dir = os.path.join(RAW_ROOT, cls)
        imgs = [f for f in os.listdir(class_raw_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]

        if not imgs:
            print(f"  [WARN] No images found for class '{cls}', skipping.")
            continue

        random.shuffle(imgs)
        n = len(imgs)

        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        n_test = n - n_train - n_val

        if n_train == 0 and n > 0:
            n_train = 1
            if n_val > 0:
                n_val -= 1
            else:
                n_test = max(n_test - 1, 0)

        train_imgs = imgs[:n_train]
        val_imgs = imgs[n_train:n_train + n_val]
        test_imgs = imgs[n_train + n_val:]

        print(f"Class '{cls}': total={n}, train={len(train_imgs)}, val={len(val_imgs)}, test={len(test_imgs)}")

        for split_name, split_list in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
            split_class_dir = os.path.join(SPLIT_ROOT, split_name, cls)
            ensure_dir(split_class_dir)

            for img_name in split_list:
                src = os.path.join(class_raw_dir, img_name)
                dst = os.path.join(split_class_dir, img_name)
                copy2(src, dst)

                rel_path = os.path.join(split_name, cls, img_name)
                index_rows.append((rel_path, cls, split_name))

    csv_path = os.path.join(SPLIT_ROOT, "dataset_index.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(index_rows)

    print(f"\n✅ Split complete! Dataset in '{SPLIT_ROOT}'")
    print(f"✅ Index CSV: {csv_path}")




if __name__ == "__main__":
    ensure_dir(RAW_ROOT)
    ensure_dir(SPLIT_ROOT)

    print("=== STEP 1: SCRAPING IMAGES (with junk filtering) ===")
    totals = {}
    for med in MEDICINES:
        totals[med] = scrape_medicine_images(med)

    print("\n=== SCRAPE SUMMARY ===")
    for med, cnt in totals.items():
        print(f"  {med:15s} -> {cnt} images")

    print("\n=== STEP 2: SPLITTING INTO TRAIN/VAL/TEST ===")
    split_dataset()

    print("\n🎉 Done!")
    print(f"Raw images:   {RAW_ROOT}/")
    print(f"Split images: {SPLIT_ROOT}/train, {SPLIT_ROOT}/val, {SPLIT_ROOT}/test")
