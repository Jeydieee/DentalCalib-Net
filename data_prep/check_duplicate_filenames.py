import json
from collections import Counter

with open(r"D:/DentalCalib-Net/data_prep/repool_output/dentex_merged_1005.json") as f:
    merged = json.load(f)

filenames = [img['file_name'] for img in merged['images']]
counts = Counter(filenames)

duplicates = {name: count for name, count in counts.items() if count > 1}

print(f"Total images in JSON: {len(filenames)}")
print(f"Unique filenames: {len(counts)}")
print(f"Duplicate filenames: {len(duplicates)}\n")

for name, count in duplicates.items():
    print(f"'{name}' appears {count} times")
    matching_ids = [img['id'] for img in merged['images'] if img['file_name'] == name]
    print(f"  image_ids: {matching_ids}")