import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

TARGET_SIZE = 640

SOURCE_DIRS = [
    r"D:/DENTEX 2023/training_data/quadrant-enumeration-disease/xrays",
    r"D:/DENTEX 2023/validation_data/quadrant_enumeration_disease/xrays",
    r"D:/DENTEX 2023/test_data/disease/input",
]

def find_source_image(file_name, source_dirs=SOURCE_DIRS):
    """Search all original DENTEX folders for this file, since the new
    repooled split doesn't match the original train/val/test locations."""
    for d in source_dirs:
        candidate = Path(d) / file_name
        if candidate.exists():
            return candidate
        candidate = Path(d) / Path(file_name).name
        if candidate.exists():
            return candidate
    return None


def resize_and_grayscale(img_path, target_size=TARGET_SIZE):
    img = cv2.imread(str(img_path))
    if img is None:
        return None, None, None

    orig_h, orig_w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_3ch = cv2.merge([gray, gray, gray])

    resized = cv2.resize(gray_3ch, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    scale_x = target_size / orig_w
    scale_y = target_size / orig_h

    return resized, scale_x, scale_y


def rescale_bbox(bbox, scale_x, scale_y):
    x, y, w, h = bbox
    return [x * scale_x, y * scale_y, w * scale_x, h * scale_y]


def process_partition(merged, images_by_id, dest_images_dir, partition_ids):
    dest_images_dir = Path(dest_images_dir)
    dest_images_dir.mkdir(parents=True, exist_ok=True)

    processed, missing = 0, []
    scale_factors = {}

    for img_id in partition_ids:
        img_info = images_by_id[img_id]
        src = find_source_image(img_info['file_name'])

        if src is None:
            missing.append(img_info['file_name'])
            continue

        resized, scale_x, scale_y = resize_and_grayscale(src)
        if resized is None:
            missing.append(img_info['file_name'])
            continue

        dest_path = dest_images_dir / Path(img_info['file_name']).name
        cv2.imwrite(str(dest_path), resized)

        scale_factors[img_id] = (scale_x, scale_y)
        processed += 1

    print(f"{dest_images_dir}: processed {processed}, missing {len(missing)}")
    if missing:
        print(f"  missing: {missing[:10]}{'...' if len(missing) > 10 else ''}")

    return scale_factors


def build_resized_json(merged, all_scale_factors, out_path='dentex_merged_1005_resized.json'):
    new_images = []
    for img in merged['images']:
        img_id = img['id']
        if img_id not in all_scale_factors:
            continue
        new_img = dict(img)
        new_img['width'] = TARGET_SIZE
        new_img['height'] = TARGET_SIZE
        new_images.append(new_img)

    new_annotations = []
    for ann in merged['annotations']:
        img_id = ann['image_id']
        if img_id not in all_scale_factors:
            continue
        scale_x, scale_y = all_scale_factors[img_id]
        new_ann = dict(ann)
        new_ann['bbox'] = rescale_bbox(ann['bbox'], scale_x, scale_y)
        new_ann['area'] = new_ann['bbox'][2] * new_ann['bbox'][3]
        new_annotations.append(new_ann)

    resized_merged = dict(merged)
    resized_merged['images'] = new_images
    resized_merged['annotations'] = new_annotations

    with open(out_path, 'w') as f:
        json.dump(resized_merged, f)
    print(f"Saved resized annotation file -> {out_path}")


if __name__ == "__main__":
    with open(r"D:/DentalCalib-Net/data_prep/repool_output/dentex_merged_1005.json") as f:
        merged = json.load(f)

    images_by_id = {img['id']: img for img in merged['images']}

    train_ids = pd.read_csv(r"D:/DentalCalib-Net/data_prep/split_output/train_ids.csv")['image_id'].tolist()
    val_ids = pd.read_csv(r"D:/DentalCalib-Net/data_prep/split_output/val_ids.csv")['image_id'].tolist()
    test_ids = pd.read_csv(r"D:/DentalCalib-Net/data_prep/split_output/test_ids.csv")['image_id'].tolist()

    all_scale_factors = {}
    all_scale_factors.update(process_partition(merged, images_by_id,
        r"D:/DentalCalib-Net/dataset_split/train", train_ids))
    all_scale_factors.update(process_partition(merged, images_by_id,
        r"D:/DentalCalib-Net/dataset_split/val", val_ids))
    all_scale_factors.update(process_partition(merged, images_by_id,
        r"D:/DentalCalib-Net/dataset_split/test", test_ids))

    build_resized_json(merged, all_scale_factors,
        out_path=r"D:/DentalCalib-Net/data_prep/repool_output/dentex_merged_1005_resized.json")