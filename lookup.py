import json
import pandas as pd

def load_everything(
    merged_path=r"D:/DentalCalib-Net/data_prep/repool_output/dentex_merged_1005.json",
    train_csv=r"D:/DentalCalib-Net/data_prep/split_output/train_ids.csv",
    val_csv=r"D:/DentalCalib-Net/data_prep/split_output/val_ids.csv",
    test_csv=r"D:/DentalCalib-Net/data_prep/split_output/test_ids.csv",
):
    with open(merged_path) as f:
        merged = json.load(f)

    images_by_id = {img['id']: img for img in merged['images']}

    partition_map = {}
    for iid in pd.read_csv(train_csv)['image_id']:
        partition_map[iid] = 'train'
    for iid in pd.read_csv(val_csv)['image_id']:
        partition_map[iid] = 'val'
    for iid in pd.read_csv(test_csv)['image_id']:
        partition_map[iid] = 'test'

    annotations_by_image = {}
    for ann in merged['annotations']:
        annotations_by_image.setdefault(ann['image_id'], []).append(ann)

    return images_by_id, partition_map, annotations_by_image


DIAG_NAMES = {0: 'Impacted', 1: 'Caries', 2: 'Periapical Lesion', 3: 'Deep Caries'}


def lookup(image_id, images_by_id, partition_map, annotations_by_image):
    if image_id not in images_by_id:
        print(f"Image ID {image_id} not found in merged pool.")
        return

    img = images_by_id[image_id]
    partition = partition_map.get(image_id, 'UNASSIGNED - not in any split!')
    anns = annotations_by_image.get(image_id, [])

    if 1 <= image_id <= 705:
        source = f"train, original id {image_id}"
    elif 706 <= image_id <= 755:
        source = f"val, original id {image_id - 705}"
    elif 756 <= image_id <= 1005:
        source = f"test, original id {image_id - 755} (verify against file_name, not this number)"
    else:
        source = "unknown range"

    print(f"\nImage ID: {image_id}")
    print(f"  file_name: {img['file_name']}")
    print(f"  partition (train/val/test split): {partition}")
    print(f"  original source: {source}")
    print(f"  size: {img['width']}x{img['height']}")
    print(f"  annotation count: {len(anns)}")

    diag_counts = {}
    for ann in anns:
        d = ann['category_id_3']
        diag_counts[d] = diag_counts.get(d, 0) + 1
    if diag_counts:
        print("  diagnoses:")
        for d, count in sorted(diag_counts.items()):
            print(f"    {DIAG_NAMES.get(d, f'unknown({d})')}: {count}")
    else:
        print("  diagnoses: none")


def search_by_filename(filename_substring, images_by_id, partition_map, annotations_by_image):
    matches = [img for img in images_by_id.values() if filename_substring in img['file_name']]
    if not matches:
        print(f"No images with filename containing '{filename_substring}'")
        return
    for img in matches:
        lookup(img['id'], images_by_id, partition_map, annotations_by_image)


if __name__ == "__main__":
    images_by_id, partition_map, annotations_by_image = load_everything()

    print(f"Loaded {len(images_by_id)} images.")
    print("Commands: type an image ID number, or 'file <name>' to search by filename, or 'quit'\n")

    while True:
        cmd = input("> ").strip()
        if cmd.lower() in ('quit', 'exit', 'q'):
            break
        elif cmd.lower().startswith('file '):
            search_by_filename(cmd[5:].strip(), images_by_id, partition_map, annotations_by_image)
        else:
            try:
                lookup(int(cmd), images_by_id, partition_map, annotations_by_image)
            except ValueError:
                print("Enter a number, 'file <name>', or 'quit'")