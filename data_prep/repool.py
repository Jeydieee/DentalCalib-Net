import json
from pathlib import Path

def load_coco(path):
    with open(path) as f:
        return json.load(f)

def remap_ids(coco, offset):
    """Shift all image_id and annotation image_id references by a fixed offset
    so IDs are globally unique across merged splits."""
    id_map = {}
    for img in coco['images']:
        old_id = img['id']
        new_id = old_id + offset
        id_map[old_id] = new_id
        img['id'] = new_id
    for ann in coco['annotations']:
        ann['image_id'] = id_map[ann['image_id']]
        ann['id'] = ann['id'] + offset * 100000  # keep annotation IDs unique too
    return coco

def merge_coco(train_coco, val_coco, test_coco):
    assert train_coco['categories_3'] == val_coco['categories_3'] == test_coco['categories_3'], \
        "Category schemas differ across splits — do not merge until resolved."

    # Remap so image IDs never collide: train stays as-is, val offset past train's max,
    # test offset past train+val's max
    train_max = max(img['id'] for img in train_coco['images'])
    val_coco = remap_ids(val_coco, offset=train_max)
    val_max = max(img['id'] for img in val_coco['images'])
    test_coco = remap_ids(test_coco, offset=val_max)

    print(f"train IDs: {min(img['id'] for img in train_coco['images'])}-{max(img['id'] for img in train_coco['images'])}")
    print(f"val IDs:   {min(img['id'] for img in val_coco['images'])}-{max(img['id'] for img in val_coco['images'])}")
    print(f"test IDs:  {min(img['id'] for img in test_coco['images'])}-{max(img['id'] for img in test_coco['images'])}")

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

    merged = merge_coco(train_coco, val_coco, test_coco)
    verify_pool(merged)

    with open(out_path, 'w') as f:
        json.dump(merged, f)
    print(f"Merged {len(merged['images'])} images -> {out_path}")

if __name__ == "__main__":
    main(
        train_path=r"C:\\Users\\mosqu\\Desktop\\DentalCalib Net\\DentalCalib-Net\\json_files\\train_quadrant_enumeration_disease.json",
        val_path=r"C:\\Users\\mosqu\\Desktop\\DentalCalib Net\\DentalCalib-Net\\json_files\\validation_triple.json",
        test_path=r"C:\\Users\\mosqu\\Desktop\\DentalCalib Net\\DentalCalib-Net\\json_files\\test_quadrant_enumeration_disease.json",
    )