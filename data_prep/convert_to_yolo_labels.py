import json
import pandas as pd
from pathlib import Path

COCO_JSON = r"D:/DentalCalib-Net/data_prep/repool_output/dentex_merged_1005_resized.json"
TRAIN_IDS_CSV = r"D:/DentalCalib-Net/data_prep/split_output/train_ids.csv"
VAL_IDS_CSV = r"D:/DentalCalib-Net/data_prep/split_output/val_ids.csv"
OUTPUT_BASE = r"D:/DentalCalib-Net/yolo_dataset"

def load_coco(path):
    with open(path) as f:
        return json.load(f)

def convert_to_yolo(merged, image_ids, out_labels_dir):
    Path(out_labels_dir).mkdir(parents=True, exist_ok=True)

    images_by_id = {img['id']: img for img in merged['images'] if img['id'] in image_ids}
    anns_by_image = {}
    for ann in merged['annotations']:
        if ann['category_id_3'] == 0 and ann['image_id'] in image_ids:  # Impacted only
            anns_by_image.setdefault(ann['image_id'], []).append(ann)

    written = 0
    for img_id, img_info in images_by_id.items():
        w, h = img_info['width'], img_info['height']
        file_name = img_info['file_name']

        label_path = Path(out_labels_dir) / (Path(file_name).stem + '.txt')

        anns = anns_by_image.get(img_id, [])
        lines = []
        for ann in anns:
            x, y, bw, bh = ann['bbox']
            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            nw = bw / w
            nh = bh / h
            lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")  # class 0 = Impacted

        with open(label_path, 'w') as f:
            f.write('\n'.join(lines))  # empty file = no impacted teeth = valid background image

        written += 1

    print(f"Wrote {written} label files -> {out_labels_dir}")
    return images_by_id


if __name__ == "__main__":
    merged = load_coco(COCO_JSON)

    train_ids = set(pd.read_csv(TRAIN_IDS_CSV)['image_id'].tolist())
    val_ids = set(pd.read_csv(VAL_IDS_CSV)['image_id'].tolist())

    convert_to_yolo(merged, train_ids, f"{OUTPUT_BASE}/labels/train")
    convert_to_yolo(merged, val_ids, f"{OUTPUT_BASE}/labels/val")