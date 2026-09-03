import json

def verify_bbox_alignment(merged_path='dentex_merged_1005.json'):
    with open(merged_path) as f:
        merged = json.load(f)

    images_by_id = {img['id']: img for img in merged['images']}

    errors = {
        'out_of_bounds': [],
        'zero_or_negative_size': [],
        'orphaned_annotation': [],
    }

    for ann in merged['annotations']:
        img_id = ann['image_id']
        img = images_by_id.get(img_id)

        if img is None:
            errors['orphaned_annotation'].append(ann['id'])
            continue

        x, y, w, h = ann['bbox']

        if w <= 0 or h <= 0:
            errors['zero_or_negative_size'].append((ann['id'], img_id, w, h))

        if x < 0 or y < 0 or (x + w) > img['width'] or (y + h) > img['height']:
            errors['out_of_bounds'].append(
                (ann['id'], img_id, (x, y, w, h), (img['width'], img['height']))
            )

    total_anns = len(merged['annotations'])
    total_errors = sum(len(v) for v in errors.values())

    print(f"Checked {total_anns} annotations.")
    print(f"Errors found: {total_errors}\n")

    for category, items in errors.items():
        print(f"{category}: {len(items)}")
        for item in items[:10]:  # show first 10 of each
            print(f"  {item}")
        if len(items) > 10:
            print(f"  ... and {len(items) - 10} more")
        print()

    return errors

if __name__ == "__main__":
    verify_bbox_alignment(r"C:/Users/JOSHUA NOGAR/Documents/DentalCalib-Net/data_prep/repool_output/dentex_merged_1005.json")