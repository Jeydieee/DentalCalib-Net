import json
import numpy as np
from scipy.stats import wilcoxon
from pathlib import Path


def cohens_d_paired(x, y):
    """Cohen's dz for paired samples: mean difference / std of differences."""
    diff = np.array(x) - np.array(y)
    if diff.std(ddof=1) == 0:
        return 0.0
    return diff.mean() / diff.std(ddof=1)


def interpret_effect_size(d):
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def load_results(path):
    with open(path) as f:
        return json.load(f)


def filter_restricted_conditions(data):
    """Keep only: clean, and S1/S2 severities of any corruption type.
    Excludes: YOLOv8 gaussian_noise S3-S5, RT-DETR gaussian_noise S2-S5
    (confirmed catastrophic-failure conditions), by simply excluding
    severity 3+ across the board and gaussian_noise S2 for RT-DETR specifically."""
    restricted = []
    for row in data:
        condition = row['condition']
        severity = row['severity']
        model = row['model']
        corruption_type = row['corruption_type']

        # keep clean always
        if condition == 'clean':
            restricted.append(row)
            continue

        # keep only severity 1-2 for OOD conditions
        if severity not in (1, 2):
            continue

        # additionally exclude RT-DETR gaussian_noise S2 specifically
        # (confirmed 0% correct starting at S2 for RT-DETR)
        if model == 'rtdetr' and corruption_type == 'gaussian_noise' and severity == 2:
            continue

        restricted.append(row)

    return restricted


def get_paired_values(data, model, metric, method_a, method_b):
    """Returns two lists of metric values, paired by condition, for two
    confidence_type methods (e.g. 'ts' vs 'dcn'), same model."""
    by_condition_a = {}
    by_condition_b = {}

    for row in data:
        if row['model'] != model:
            continue
        if row['confidence_type'] == method_a:
            by_condition_a[row['condition']] = row[metric]
        elif row['confidence_type'] == method_b:
            by_condition_b[row['condition']] = row[metric]

    shared_conditions = sorted(set(by_condition_a.keys()) & set(by_condition_b.keys()))

    values_a = [by_condition_a[c] for c in shared_conditions]
    values_b = [by_condition_b[c] for c in shared_conditions]

    return values_a, values_b, shared_conditions


def run_restricted_comparison(data, models, metrics, method_pairs):
    results = []

    for model in models:
        for metric in metrics:
            for method_a, method_b in method_pairs:
                values_a, values_b, conditions = get_paired_values(data, model, metric, method_a, method_b)

                n = len(conditions)
                if n < 3:
                    results.append({
                        'model': model,
                        'metric': metric,
                        'comparison': f"{method_a}_vs_{method_b}",
                        'n_conditions': n,
                        'note': 'insufficient paired conditions for test',
                    })
                    continue

                try:
                    stat, p = wilcoxon(values_a, values_b)
                except ValueError as e:
                    results.append({
                        'model': model,
                        'metric': metric,
                        'comparison': f"{method_a}_vs_{method_b}",
                        'n_conditions': n,
                        'note': f'wilcoxon failed: {e}',
                    })
                    continue

                d = cohens_d_paired(values_a, values_b)

                results.append({
                    'model': model,
                    'metric': metric,
                    'comparison': f"{method_a}_vs_{method_b}",
                    'n_conditions': n,
                    'conditions_used': conditions,
                    'mean_a': float(np.mean(values_a)),
                    'mean_b': float(np.mean(values_b)),
                    'wilcoxon_stat': float(stat),
                    'p_value': float(p),
                    'cohens_dz': float(d),
                    'effect_size': interpret_effect_size(d),
                })

    return results


if __name__ == "__main__":
    RESULTS_PATH = r"D:\DentalCalib-Net\stage5_outputs\calibration_results.json"
    OUTPUT_PATH = r"D:\DentalCalib-Net\stage5_outputs\statistical_tests_restricted.json"

    data = load_results(RESULTS_PATH)
    print(f"Original rows: {len(data)}")

    restricted_data = filter_restricted_conditions(data)
    print(f"Restricted rows (clean + S1-S2, excluding RT-DETR gaussian_noise S2): {len(restricted_data)}")

    conditions_included = sorted(set(row['condition'] for row in restricted_data))
    print(f"Conditions included: {conditions_included}")

    MODELS = ['yolov8', 'rtdetr']
    METRICS = ['ece', 'mce', 'dece', 'rmsd']
    METHOD_PAIRS = [('raw', 'ts'), ('raw', 'dcn'), ('ts', 'dcn')]

    results = run_restricted_comparison(restricted_data, MODELS, METRICS, METHOD_PAIRS)

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{len(results)} comparisons computed. Saved -> {OUTPUT_PATH}")
    print("\nSummary (significant results, p<0.05):")
    for r in results:
        if 'p_value' in r and r['p_value'] < 0.05:
            print(f"  {r['model']} {r['metric']} {r['comparison']}: "
                  f"mean_a={r['mean_a']:.4f}, mean_b={r['mean_b']:.4f}, "
                  f"p={r['p_value']:.4f}, d={r['cohens_dz']:.2f} ({r['effect_size']})")