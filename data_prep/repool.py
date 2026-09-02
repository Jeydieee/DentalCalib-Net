import json
from pathlib import Path

def load_coco(path):
    with open(path) as f:
        return json.load(f)

def check_id_collisions(train_coco, val_coco, test_coco):
    train_ids = {img['id'] for img in train_coco['images']}
    val_ids   = {img['id'] for img in val_coco['images']}
    test_ids  = {img['id'] for img in test_coco['images']}
    overlaps = (
        len(train_ids & val_ids),
        len(train_ids & test_ids),
        len(val_ids & test_ids),
    )
    if any(overlaps):
        raise ValueError(f"ID collisions found (train-val, train-test, val-test): {overlaps}. "
                          f"Remap IDs before merging.")
    return True

def merge_coco(train_coco, val_coco, test_coco):
    assert train_coco['categories_3'] == val_coco['categories_3'] == test_coco['categories_3'], \
        "Category schemas differ across splits — do not merge until resolved."
    merged = {
        'images': train_coco['images'] + val_coco['images'] + test_coco['images'],
        'annotations': train_coco['annotations'] + val_coco['annotations'] + test_coco['annotations'],
        'categories_1': train_coco['categories_1'],
        'categories_2': train_coco['categories_2'],
        'categories_3': train_coco['categories_3'],
    }
    return merged

def verify_pool(merged, expected_n=1005):
    n_images = len(merged['images'])
    n_unique = len(set(img['id'] for img in merged['images']))
    assert n_images == expected_n, f"Expected {expected_n} images, got {n_images}"
    assert n_unique == expected_n, f"Duplicate IDs found: {n_images - n_unique} duplicates"
    return True

def main(train_path, val_path, test_path, out_path='dentex_merged_1005.json'):
    train_coco = load_coco(train_path)
    val_coco   = load_coco(val_path)
    test_coco  = load_coco(test_path)

    check_id_collisions(train_coco, val_coco, test_coco)
    merged = merge_coco(train_coco, val_coco, test_coco)
    verify_pool(merged)

    with open(out_path, 'w') as f:
        json.dump(merged, f)
    print(f"Merged {len(merged['images'])} images -> {out_path}")

if __name__ == "__main__":
    main(
        train_path="D:\\DentalCalib-Net\\json_files\\train_quadrant_enumeration_disease.json",
        val_path="D:\\DentalCalib-Net\\json_files\\validation_triple.json",
        test_path="D:\\DentalCalib-Net\\json_files\\test_quadrant_enumeration_disease.json",
    )