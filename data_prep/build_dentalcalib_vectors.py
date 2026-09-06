import json
from pathlib import Path

def normalize_bbox(bbox_xyxy, img_width=640, img_height=640):
    """Convert [x1,y1,x2,y2] to normalized [width, height, center_x, center_y],
    each scaled to [0,1] relative to image dimensions."""
    x1, y1, x2, y2 = bbox_xyxy
    w = (x2 - x1) / img_width
    h = (y2 - y1) / img_height
    cx = ((x1 + x2) / 2) / img_width
    cy = ((y1 + y2) / 2) / img_height
    return [w, h, cx, cy]


def build_vectors(labeled_predictions_path, quality_features, quality_key, out_path):
    """Combines labeled predictions with quality features into 8-dim vectors.
    quality_key: which key in quality_features['val'] etc. to use per image."""
    with open(labeled_predictions_path) as f:
        labeled_data = json.load(f)

    vectors = []
    skipped_no_quality = 0

    for entry in labeled_data:
        filename = Path(entry['image_path']).name

        img_quality = quality_features.get(quality_key, {}).get(filename)
        if img_quality is None:
            skipped_no_quality += 1
            continue

        for pred in entry['predictions']:
            bbox_features = normalize_bbox(pred['bbox_xyxy'])
            confidence = pred['confidence']
            label = pred['label']

            input_vector = [
                confidence,
                bbox_features[0],  # width
                bbox_features[1],  # height
                bbox_features[2],  # center_x
                bbox_features[3],  # center_y
                img_quality['laplacian_variance'],
                img_quality['pixel_std_dev'],
                img_quality['jpeg_block_artifact_score'],
            ]

            vectors.append({
                'image_filename': filename,
                'input_vector': input_vector,
                'label': label,
            })

    with open(out_path, 'w') as f:
        json.dump(vectors, f)

    print(f"{Path(labeled_predictions_path).name}: {len(vectors)} vectors built, "
          f"{skipped_no_quality} images skipped (no quality features found) -> {out_path}")

    return vectors


if __name__ == "__main__":
    LABELED_DIR = r"D:\DentalCalib-Net\stage4_outputs\labeled_predictions"
    QUALITY_FEATURES_PATH = r"D:\DentalCalib-Net\data_prep\quality_features.json"
    OUTPUT_DIR = r"D:\DentalCalib-Net\stage4_outputs\training_vectors"

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    with open(QUALITY_FEATURES_PATH) as f:
        quality_features = json.load(f)

    # YOLOv8 validation vectors -- quality_key 'val' since these are validation images
    yolo_vectors = build_vectors(
        f"{LABELED_DIR}/yolov8_predictions_val_labeled.json",
        quality_features,
        'val',
        f"{OUTPUT_DIR}/yolov8_val_vectors.json",
    )

    # RT-DETR validation vectors (filtered, top-50)
    rtdetr_vectors = build_vectors(
        f"{LABELED_DIR}/rtdetr_predictions_val_filtered_labeled.json",
        quality_features,
        'val',
        f"{OUTPUT_DIR}/rtdetr_val_vectors.json",
    )

    # quick class balance check
    def report_balance(vectors, name):
        total = len(vectors)
        positive = sum(1 for v in vectors if v['label'] == 1)
        pct = 100 * positive / total if total > 0 else 0
        print(f"{name}: {total} total, {positive} positive ({pct:.1f}%)")

    print()
    report_balance(yolo_vectors, "YOLOv8 training vectors")
    report_balance(rtdetr_vectors, "RT-DETR training vectors")