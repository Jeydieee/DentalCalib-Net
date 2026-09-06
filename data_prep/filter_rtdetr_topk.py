import json
from pathlib import Path

def filter_top_k(predictions_data, k=50):
    """Keep only the top-K highest-confidence predictions per image."""
    filtered = []
    for entry in predictions_data:
        sorted_preds = sorted(entry['predictions'], key=lambda p: p['confidence'], reverse=True)
        top_k_preds = sorted_preds[:k]
        filtered.append({
            'image_path': entry['image_path'],
            'predictions': top_k_preds,
        })
    return filtered


def process_and_filter(pred_path, out_path, k=50):
    with open(pred_path) as f:
        predictions_data = json.load(f)

    original_count = sum(len(e['predictions']) for e in predictions_data)
    filtered_data = filter_top_k(predictions_data, k=k)
    filtered_count = sum(len(e['predictions']) for e in filtered_data)

    with open(out_path, 'w') as f:
        json.dump(filtered_data, f)

    print(f"{Path(pred_path).name}: {original_count} -> {filtered_count} predictions -> {out_path}")


if __name__ == "__main__":
    STAGE2_DIR = r"D:/DentalCalib-Net/stage2_outputs"
    OUTPUT_DIR = r"D:/DentalCalib-Net/stage4_outputs/rtdetr_filtered"
    K = 50

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    CORRUPTION_TYPES = ['gaussian_noise', 'motion_blur', 'brightness_variation', 'jpeg_compression']
    SEVERITIES = ['S1', 'S2', 'S3', 'S4', 'S5']

    print("Filtering RT-DETR clean predictions...")
    process_and_filter(
        f"{STAGE2_DIR}/rtdetr_predictions_clean.json",
        f"{OUTPUT_DIR}/rtdetr_predictions_clean_filtered.json",
        k=K,
    )

    print("\nFiltering RT-DETR OOD predictions...")
    for corruption in CORRUPTION_TYPES:
        for severity in SEVERITIES:
            pred_path = f"{STAGE2_DIR}/rtdetr_ood_predictions/rtdetr_{corruption}_{severity}.json"
            out_path = f"{OUTPUT_DIR}/rtdetr_{corruption}_{severity}_filtered.json"
            process_and_filter(pred_path, out_path, k=K)

    # NEW: also filter the validation set predictions
    print("\nFiltering RT-DETR validation predictions...")
    process_and_filter(
        f"{STAGE2_DIR}/rtdetr_predictions_val.json",
        f"{OUTPUT_DIR}/rtdetr_predictions_val_filtered.json",
        k=K,
    )

    print("\nDone.")