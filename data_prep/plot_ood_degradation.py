import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_calibration_results(path):
    with open(path) as f:
        return json.load(f)


def load_statistical_tests(path):
    with open(path) as f:
        return json.load(f)


def get_spearman_result(stat_tests, model, confidence_type):
    """Find the aggregate (across all 21 conditions) Spearman rho + Bonferroni-
    corrected significance for a given model/confidence_type from Family C."""
    for entry in stat_tests:
        if (entry.get('family') == 'C_map_vs_ece_spearman' and
            entry.get('model') == model and
            entry.get('confidence_type') == confidence_type):
            return entry.get('spearman_rho'), entry.get('p_bonferroni'), entry.get('significant_bonferroni_0.05')
    return None, None, None


def classify_coupling(rho, significant):
    """Per the thesis proposal's Definition of Terms: coupled = significant
    negative correlation. Decoupled/dangerous = non-significant, near-zero,
    or positive."""
    if rho is None or significant is None:
        return "unknown"
    if significant and rho < 0:
        return "coupled"
    else:
        return "decoupled_dangerous"


def get_condition_order(calibration_data, model, confidence_type):
    """Fixed, consistent ordering: clean first, then each corruption type's
    5 severities in sequence."""
    CORRUPTION_TYPES = ['gaussian_noise', 'motion_blur', 'brightness_variation', 'jpeg_compression']
    SEVERITIES = ['S1', 'S2', 'S3', 'S4', 'S5']

    ordered_conditions = ['clean']
    for c in CORRUPTION_TYPES:
        for s in SEVERITIES:
            ordered_conditions.append(f"{c}_{s}")

    ece_values = []
    map_values = []
    labels = []

    for condition in ordered_conditions:
        row = next((r for r in calibration_data
                    if r['model'] == model
                    and r['condition'] == condition
                    and r['confidence_type'] == confidence_type), None)
        if row is None:
            ece_values.append(np.nan)
            map_values.append(np.nan)
        else:
            ece_values.append(row['ece'])
            map_values.append(row['map50'])
        labels.append(condition)

    return labels, ece_values, map_values


def plot_full_degradation(calibration_data, model, confidence_type, rho, significant, output_dir):
    labels, ece_values, map_values = get_condition_order(calibration_data, model, confidence_type)

    fig, ax1 = plt.subplots(figsize=(14, 5))

    x = range(len(labels))
    ax1.set_xlabel('Condition (Clean, then 4 corruption types x 5 severities)')
    ax1.set_ylabel('ECE', color='tab:red')
    ax1.plot(x, ece_values, color='tab:red', marker='o', markersize=4, label='ECE')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=90, fontsize=7)

    ax2 = ax1.twinx()
    ax2.set_ylabel('mAP@0.50', color='tab:blue')
    ax2.plot(x, map_values, color='tab:blue', marker='s', markersize=4, label='mAP@0.50')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    coupling_status = classify_coupling(rho, significant)
    status_label = {
        "coupled": "Coupled (calibration tracks accuracy)",
        "decoupled_dangerous": "Decoupled - Dangerous (calibration independent of accuracy)",
        "unknown": "Status unknown (missing statistical test data)",
    }[coupling_status]

    rho_str = f"{rho:.3f}" if rho is not None else "N/A"

    title = (f"{model.upper()} — {confidence_type.upper()} — All 21 Conditions\n"
             f"Spearman ρ (aggregate, Bonferroni-corrected) = {rho_str} — {status_label}")
    ax1.set_title(title, fontsize=11)

    fig.tight_layout()

    out_path = f"{output_dir}/{model}_{confidence_type}_full_degradation.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"{model} / {confidence_type}: rho={rho_str}, significant={significant} -> {coupling_status} -> {out_path}")


if __name__ == "__main__":
    CALIBRATION_RESULTS_PATH = r"D:\DentalCalib-Net\stage5_outputs\calibration_results.json"
    STAT_TESTS_PATH = r"D:\DentalCalib-Net\stage5_outputs\statistical_tests.json"
    OUTPUT_DIR = r"D:\DentalCalib-Net\stage5_outputs\ood_degradation_plots"

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    calibration_data = load_calibration_results(CALIBRATION_RESULTS_PATH)
    stat_tests = load_statistical_tests(STAT_TESTS_PATH)

    MODELS = ['yolov8', 'rtdetr']
    CONFIDENCE_TYPES = ['raw', 'ts', 'dcn']

    for model in MODELS:
        for confidence_type in CONFIDENCE_TYPES:
            rho, p_bonf, significant = get_spearman_result(stat_tests, model, confidence_type)
            plot_full_degradation(calibration_data, model, confidence_type, rho, significant, OUTPUT_DIR)

    print("\nDone. 6 plots generated.")