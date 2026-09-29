REPOOL_JSON = r"D:\DentalCalib-Net\data_prep\repool_output\dentex_merged_1005_resized.json"
TEST_IDS_CSV = r"D:\DentalCalib-Net\data_prep\split_output\test_ids.csv"

import json
import csv

with open(REPOOL_JSON) as f:
    data = json.load(f)

id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}

with open(TEST_IDS_CSV) as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    test_ids = [int(row[0]) for row in reader if row]

print(f"{len(test_ids)} test IDs loaded")
print("\nFirst 5 test IDs, mapped:")
for tid in test_ids[:5]:
    fname = id_to_filename.get(tid, "NOT FOUND")
    print(f"  image_id={tid} -> {fname}")

found = sum(1 for tid in test_ids if tid in id_to_filename)
print(f"\n{found}/{len(test_ids)} test IDs found in repool mapping")