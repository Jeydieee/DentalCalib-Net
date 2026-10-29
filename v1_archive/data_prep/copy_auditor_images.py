try:
    from .paths import project_path, raw_data_path
except ImportError:
    from paths import project_path, raw_data_path

REPOOL_JSON = project_path("data_prep", "repool_output", "dentex_merged_1005_resized.json")
TEST_IDS_CSV = project_path("data_prep", "split_output", "test_ids.csv")
DATASET_SPLIT_DIR = project_path("dataset_split")
OUT_DIR = project_path("frontend", "images")

import json
import csv
import shutil
from pathlib import Path

CORRUPTIONS = ["gaussian_noise", "motion_blur", "brightness_variation", "jpeg_compression"]
SEVERITIES = [1, 2, 3, 4, 5]

with open(REPOOL_JSON) as f:
    data = json.load(f)
id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}

with open(TEST_IDS_CSV) as f:
    reader = csv.reader(f)
    next(reader)
    test_ids = [int(row[0]) for row in reader if row]

test_filenames = [id_to_filename[tid] for tid in test_ids if tid in id_to_filename]
print(f"Resolved {len(test_filenames)}/{len(test_ids)} test image filenames")

out_root = Path(OUT_DIR)
out_root.mkdir(parents=True, exist_ok=True)


def copy_set(src_dir, dst_dir):
    dst_dir.mkdir(exist_ok=True)
    copied, missing = 0, []
    for fname in test_filenames:
        src = Path(src_dir) / fname
        if not src.exists():
            missing.append(fname)
            continue
        shutil.copy2(src, dst_dir / fname)
        copied += 1
    return copied, missing


# --- Clean images ---
copied, missing = copy_set(Path(DATASET_SPLIT_DIR) / "test", out_root / "clean")
print(f"clean: copied {copied}/{len(test_filenames)}" + (f", missing e.g. {missing[:3]}" if missing else ""))

# --- OOD corrupted images ---
for corruption in CORRUPTIONS:
    for sev in SEVERITIES:
        src_dir = Path(DATASET_SPLIT_DIR) / "ood" / corruption / f"S{sev}"
        copied, missing = copy_set(src_dir, out_root / f"{corruption}_S{sev}")
        print(f"{corruption}_S{sev}: copied {copied}/{len(test_filenames)}" +
              (f", missing e.g. {missing[:3]}" if missing else ""))

print("\nDone.")