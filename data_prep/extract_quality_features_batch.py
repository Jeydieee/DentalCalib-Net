import cv2
import numpy as np
import json
from pathlib import Path

# ── Quality feature functions (same as before) ──────────────────────

def laplacian_variance(img):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def pixel_std_dev(img):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    return float(gray.std())


def jpeg_block_artifact_score(img, block_size=8):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    gray = gray.astype(np.float64)
    h, w = gray.shape

    diffs = []
    for x in range(block_size, w, block_size):
        if x >= w:
            break
        diffs.append(np.mean(np.abs(gray[:, x - 1] - gray[:, x])))

    for y in range(block_size, h, block_size):
        if y >= h:
            break
        diffs.append(np.mean(np.abs(gray[y - 1, :] - gray[y, :])))

    if not diffs:
        return 0.0
    return float(np.mean(diffs))


def extract_quality_features(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    return {
        'laplacian_variance': laplacian_variance(img),
        'pixel_std_dev': pixel_std_dev(img),
        'jpeg_block_artifact_score': jpeg_block_artifact_score(img),
    }


# ── Batch extraction ─────────────────────────────────────────────────

VAL_DIR = r"D:/DentalCalib-Net/dataset_split/val"
TEST_DIR = r"D:/DentalCalib-Net/dataset_split/test"
OOD_BASE = r"D:/DentalCalib-Net/dataset_split/ood"
OUTPUT_PATH = r"D:/DentalCalib-Net/data_prep/quality_features.json"

CORRUPTION_TYPES = ['gaussian_noise', 'motion_blur', 'brightness_variation', 'jpeg_compression']
SEVERITIES = ['S1', 'S2', 'S3', 'S4', 'S5']


def process_folder(folder_path):
    """Returns dict of {filename: features} for all PNGs in a folder."""
    folder = Path(folder_path)
    results = {}
    missing = []

    for img_path in sorted(folder.glob('*.png')):
        features = extract_quality_features(img_path)
        if features is None:
            missing.append(img_path.name)
            continue
        results[img_path.name] = features

    if missing:
        print(f"  WARNING: {len(missing)} images failed to load in {folder_path}")

    return results


def main():
    output = {
        "val": {},
        "test_clean": {},
        "test_ood": {},
    }

    print("Processing validation set...")
    output["val"] = process_folder(VAL_DIR)
    print(f"  {len(output['val'])} images processed.")

    print("Processing clean test set...")
    output["test_clean"] = process_folder(TEST_DIR)
    print(f"  {len(output['test_clean'])} images processed.")

    print("Processing OOD test sets...")
    for corruption in CORRUPTION_TYPES:
        for severity in SEVERITIES:
            key = f"{corruption}_{severity}"
            folder = f"{OOD_BASE}/{corruption}/{severity}"
            output["test_ood"][key] = process_folder(folder)
            print(f"  {key}: {len(output['test_ood'][key])} images processed.")

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f)

    print(f"\nDone. Saved -> {OUTPUT_PATH}")

    # summary counts
    total = len(output["val"]) + len(output["test_clean"])
    total += sum(len(v) for v in output["test_ood"].values())
    print(f"Total images processed: {total}")


if __name__ == "__main__":
    main()