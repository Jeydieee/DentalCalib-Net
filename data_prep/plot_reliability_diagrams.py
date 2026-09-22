import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def compute_reliability_bins(confidences, labels, n_bins=15):
    confidences = np.array(confidences)
    labels = np.array(labels)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_midpoints = []
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (confidences >= bin_edges[i]) & (confidences <= bin_edges[i + 1])
        else:
            mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])

        if mask.sum() == 0:
            continue

        bin_midpoints.append((bin_edges[i] + bin_edges[i + 1]) / 2)
        bin_accuracies.append(labels[mask].mean())
        bin_confidences.append(confidences[mask].mean())
        bin_counts.append(int(mask.sum()))

    return bin_midpoints, bin_accuracies, bin_confidences, bin_counts


def load_json(path):
    with open(path) as f:
        return json.load(f)


def merge_labeled_and_recalibrated(labeled_data, recalibrated_data):
    """Merge by image_path + prediction index, same approach used for
    Group A's calibration metrics. Returns a flat list of merged prediction dicts."""
    merged = []

    labeled_by_image = {entry['image_path']: entry['predictions'] for entry in labeled_data}
    recal_by_image = {entry['image_path']: entry['predictions'] for entry in recalibrated_data}

    for image_path, recal_preds in recal_by_image.items():
        labeled_preds = labeled_by_image.get(image_path)
        if labeled_preds is None or len(labeled_preds) != len(recal_preds):
            continue  # skip mismatched images rather than risk misaligned data

        for i in range(len(recal_preds)):
            combined = dict(recal_preds[i])
            combined['label'] = labeled_preds[i]['label']
            combined['best_iou'] = labeled_preds[i].get('best_iou')
            merged.append(combined)

    return merged


def extract_confidences_labels(merged_predictions, confidence_field):
    confidences = [p[confidence_field] for p in merged_predictions]
    labels = [p['label'] for p in merged_predictions]
    return confidences, labels


def plot_reliability_diagram(confidences, labels, title, output_path, n_bins=15):
    bin_midpoints, bin_accuracies, bin_confidences, bin_counts = compute_reliability_bins(
        confidences, labels, n_bins
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect calibration')

    if bin_midpoints:
        ax.plot(bin_confidences, bin_accuracies, marker='o', color='tab:red', label='Observed')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Mean Predicted Confidence')
    ax.set_ylabel('Observed Accuracy')
    ax.set_title(title, fontsize=10)
    ax.legend()

    total_preds = len(confidences)
    n_populated_bins = len(bin_midpoints)
    ax.text(0.05, 0.92, f"n={total_preds} predictions\n{n_populated_bins}/{n_bins} bins populated",
            transform=ax.transAxes, fontsize=8, verticalalignment='top')

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def find_labeled_path(labeled_dir, model, condition):
    """RT-DETR uses '_filtered_labeled.json' for its labeled files;
    YOLOv8 uses plain '_labeled.json'. Clean vs OOD also has slightly
    different naming, matching what earlier scripts in this project used."""
    if model == 'rtdetr':
        if condition == 'clean':
            candidate = f"{labeled_dir}/rtdetr_predictions_clean_filtered_labeled.json"
        else:
            candidate = f"{labeled_dir}/rtdetr_{condition}_filtered_labeled.json"
    else:
        if condition == 'clean':
            candidate = f"{labeled_dir}/yolov8_predictions_clean_labeled.json"
        else:
            candidate = f"{labeled_dir}/yolov8_{condition}_labeled.json"

    return candidate if Path(candidate).exists() else None


def generate_all_reliability_diagrams(recalibrated_dir, labeled_dir, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    MODELS = ['yolov8', 'rtdetr']
    CORRUPTION_TYPES = ['gaussian_noise', 'motion_blur', 'brightness_variation', 'jpeg_compression']
    SEVERITIES = ['S1', 'S2', 'S3', 'S4', 'S5']
    CONFIDENCE_FIELDS = {'raw': 'confidence_raw', 'ts': 'confidence_ts', 'dcn': 'confidence_dcn'}

    conditions = ['clean'] + [f"{c}_{s}" for c in CORRUPTION_TYPES for s in SEVERITIES]

    generated = 0
    skipped = 0

    for model in MODELS:
        for condition in conditions:
            recal_path = f"{recalibrated_dir}/{model}_{condition}_recalibrated.json"
            if not Path(recal_path).exists():
                skipped += 1
                continue

            labeled_path = find_labeled_path(labeled_dir, model, condition)
            if labeled_path is None:
                print(f"  WARNING: no labeled file found for {model} {condition}, skipping")
                skipped += 1
                continue

            recalibrated_data = load_json(recal_path)
            labeled_data = load_json(labeled_path)

            merged = merge_labeled_and_recalibrated(labeled_data, recalibrated_data)

            if len(merged) == 0:
                skipped += 1
                continue

            for method_key, confidence_field in CONFIDENCE_FIELDS.items():
                confidences, labels = extract_confidences_labels(merged, confidence_field)

                if len(confidences) == 0:
                    skipped += 1
                    continue

                title = f"{model.upper()} — {condition} — {method_key.upper()}"
                out_path = f"{output_dir}/{model}_{condition}_{method_key}_reliability.png"
                plot_reliability_diagram(confidences, labels, title, out_path)
                generated += 1

    print(f"\nGenerated {generated} reliability diagrams, skipped {skipped}")


if __name__ == "__main__":
    RECALIBRATED_DIR = r"D:\DentalCalib-Net\stage4_outputs\recalibrated_predictions"
    LABELED_DIR = r"D:\DentalCalib-Net\stage4_outputs\labeled_predictions"
    OUTPUT_DIR = r"D:\DentalCalib-Net\stage5_outputs\reliability_diagrams"

    generate_all_reliability_diagrams(RECALIBRATED_DIR, LABELED_DIR, OUTPUT_DIR)
    print("\nDone.")