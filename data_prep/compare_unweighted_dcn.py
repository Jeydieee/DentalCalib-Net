import json
import numpy as np

def load_results(path):
    with open(path) as f:
        return json.load(f)

def aggregate_by_confidence_type(results, metric):
    """Average a metric across all conditions/models, grouped by confidence_type."""
    out = {}
    for r in results:
        ct = r["confidence_type"]
        val = r.get(metric)
        if val is None:
            continue
        out.setdefault(ct, []).append(val)
    return {ct: float(np.mean(vals)) for ct, vals in out.items()}

def aggregate_by_model_and_type(results, metric):
    out = {}
    for r in results:
        key = (r["model"], r["confidence_type"])
        val = r.get(metric)
        if val is None:
            continue
        out.setdefault(key, []).append(val)
    return {k: float(np.mean(vals)) for k, vals in out.items()}


if __name__ == "__main__":
    ORIGINAL_PATH = r"D:\DentalCalib-Net\stage5_outputs\calibration_results.json"
    UNWEIGHTED_PATH = r"D:\DentalCalib-Net\stage5_outputs\calibration_results_unweighted.json"

    original = load_results(ORIGINAL_PATH)
    unweighted = load_results(UNWEIGHTED_PATH)

    metrics = ["ece", "mce", "dece", "rmsd"]

    print("=" * 70)
    print("AGGREGATE COMPARISON (mean across all 21 conditions, both models)")
    print("=" * 70)

    for metric in metrics:
        orig_agg = aggregate_by_confidence_type(original, metric)
        unw_agg = aggregate_by_confidence_type(unweighted, metric)

        print(f"\n--- {metric.upper()} ---")
        print(f"  raw                    : {orig_agg.get('raw', float('nan')):.4f}")
        print(f"  temperature scaling    : {orig_agg.get('ts', float('nan')):.4f}")
        print(f"  DCN (class-weighted)   : {orig_agg.get('dcn', float('nan')):.4f}")
        print(f"  DCN (unweighted)       : {unw_agg.get('dcn', float('nan')):.4f}")

    print("\n" + "=" * 70)
    print("PER-MODEL BREAKDOWN")
    print("=" * 70)

    for metric in metrics:
        orig_pm = aggregate_by_model_and_type(original, metric)
        unw_pm = aggregate_by_model_and_type(unweighted, metric)

        print(f"\n--- {metric.upper()} ---")
        for model in ["yolov8", "rtdetr"]:
            print(f"  [{model}]")
            print(f"    raw                  : {orig_pm.get((model, 'raw'), float('nan')):.4f}")
            print(f"    temperature scaling  : {orig_pm.get((model, 'ts'), float('nan')):.4f}")
            print(f"    DCN (class-weighted) : {orig_pm.get((model, 'dcn'), float('nan')):.4f}")
            print(f"    DCN (unweighted)     : {unw_pm.get((model, 'dcn'), float('nan')):.4f}")

    print("\n" + "=" * 70)