import json

def compute_iou(box1, box2):
    """box format: [x1, y1, x2, y2]. Returns IoU as a float."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def bbox_xywh_to_xyxy(bbox):
    """Ground truth boxes are stored as [x, y, w, h] -- convert to [x1,y1,x2,y2]."""
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def load_ground_truth(json_path):
    """Returns dict: {file_name: [list of gt boxes in xyxy, category_id_3 == 0 only]}."""
    with open(json_path) as f:
        merged = json.load(f)

    images_by_id = {img['id']: img['file_name'] for img in merged['images']}

    gt_by_filename = {}
    for ann in merged['annotations']:
        if ann['category_id_3'] != 0:  # Impacted only
            continue
        file_name = images_by_id.get(ann['image_id'])
        if file_name is None:
            continue
        gt_box = bbox_xywh_to_xyxy(ann['bbox'])
        gt_by_filename.setdefault(file_name, []).append(gt_box)

    return gt_by_filename
  
if __name__ == "__main__":
    GT_PATH = r"D:/DentalCalib-Net/data_prep/repool_output/dentex_merged_1005_resized.json"

    gt_by_filename = load_ground_truth(GT_PATH)

    # pick one known image to sanity-check
    test_image = "test_0.png"

    if test_image in gt_by_filename:
        print(f"Ground truth boxes for {test_image}:")
        for box in gt_by_filename[test_image]:
            print(" ", box)
    else:
        print(f"{test_image} has no impacted-tooth ground truth (0 boxes) or wasn't found.")

    print(f"\nTotal images with at least one ground-truth impacted box: {len(gt_by_filename)}")

    # quick IoU sanity check with made-up boxes
    box_a = [100, 100, 200, 200]
    box_b = [150, 150, 250, 250]
    print(f"\nIoU test (overlapping boxes): {compute_iou(box_a, box_b):.4f}")

    box_c = [300, 300, 400, 400]  # no overlap with box_a
    print(f"IoU test (non-overlapping boxes): {compute_iou(box_a, box_c):.4f}")