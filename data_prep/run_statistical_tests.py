from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT_DIR = Path(__file__).resolve().parent.parent

IN_CSV = ROOT_DIR / "stage5_outputs/calibration_results.csv"
OUT_JSON = ROOT_DIR / "stage5_outputs/statistical_tests.json"
OUT_CSV = ROOT_DIR / "stage5_outputs/statistical_tests.csv"

MODELS = ["yolov8", "rtdetr"]
CONF_TYPES = ["raw", "ts", "dcn"]
METRICS = ["ece", "mce", "dece", "rmsd"]

CONDITIONS = ["clean"] + [
    f"{corr}_S{sev}"
    for corr in ("brightness_variation", "gaussian_noise",
                 "jpeg_compression", "motion_blur")
    for sev in (1, 2, 3, 4, 5)
]


def cohens_dz(diff):
    """Paired/one-sample effect size: mean difference in units of its own std."""
    diff = np.asarray(diff, dtype=np.float64)
    sd = diff.std(ddof=1)
    if sd == 0:
        return 0.0 if diff.mean() == 0 else float("inf") * np.sign(diff.mean())
    return float(diff.mean() / sd)


def cohens_d_pooled(x, y):
    """Independent-groups effect size, pooled standard deviation."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    nx, ny = x.size, y.size
    pooled_sd = np.sqrt(
        ((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2)
    )
    if pooled_sd == 0:
        return 0.0 if x.mean() == y.mean() else float("inf") * np.sign(x.mean() - y.mean())
    return float((x.mean() - y.mean()) / pooled_sd)


def bonferroni(rows, p_key="p_value", n=None):
    n = n if n is not None else len(rows)
    for r in rows:
        r["p_bonferroni"] = min(r[p_key] * n, 1.0)
        r["significant_bonferroni_0.05"] = r["p_bonferroni"] < 0.05
    return rows


def get_series(df, model, conf_type, metric):
    sub = df[(df["model"] == model) & (df["confidence_type"] == conf_type)]
    sub = sub.set_index("condition").reindex(CONDITIONS)
    if sub[metric].isnull().any():
        missing = sub[sub[metric].isnull()].index.tolist()
        raise ValueError(
            f"missing {metric} for model={model} conf={conf_type}, "
            f"conditions={missing}"
        )
    return sub[metric].to_numpy(dtype=np.float64)


# ── Group A: DCN vs TS vs Raw, paired within model across the 21 conditions ──

def group_a_dcn_vs_ts_vs_raw(df):
    rows = []
    pairs = [("raw", "ts"), ("raw", "dcn"), ("ts", "dcn")]
    for model in MODELS:
        for metric in METRICS:
            for a, b in pairs:
                xa = get_series(df, model, a, metric)
                xb = get_series(df, model, b, metric)
                diff = xa - xb
                if np.allclose(diff, 0.0):
                    stat, p = 0.0, 1.0
                else:
                    stat, p = stats.wilcoxon(xa, xb)
                rows.append({
                    "family": "A_dcn_vs_ts_vs_raw",
                    "model": model,
                    "metric": metric,
                    "compare": f"{a}_vs_{b}",
                    "n_pairs": int(len(xa)),
                    "mean_a": float(xa.mean()),
                    "mean_b": float(xb.mean()),
                    "mean_diff_a_minus_b": float(diff.mean()),
                    "wilcoxon_stat": float(stat),
                    "p_value": float(p),
                    "cohens_dz": cohens_dz(diff),
                })
    return bonferroni(rows, n=len(rows))


# ── Group B: YOLOv8 vs RT-DETR, independent groups across 21 conditions ──

def group_b_yolov8_vs_rtdetr(df):
    rows = []
    for metric in METRICS:
        for conf_type in CONF_TYPES:
            x = get_series(df, "yolov8", conf_type, metric)
            y = get_series(df, "rtdetr", conf_type, metric)
            stat, p = stats.mannwhitneyu(x, y, alternative="two-sided")
            rows.append({
                "family": "B_yolov8_vs_rtdetr",
                "metric": metric,
                "confidence_type": conf_type,
                "n_yolov8": int(len(x)),
                "n_rtdetr": int(len(y)),
                "mean_yolov8": float(x.mean()),
                "mean_rtdetr": float(y.mean()),
                "mannwhitney_u": float(stat),
                "p_value": float(p),
                "cohens_d_pooled": cohens_d_pooled(x, y),
            })
    return bonferroni(rows, n=len(rows))


# ── Group C: mAP vs ECE Spearman correlation, per model per confidence type ──

def group_c_map_vs_ece_spearman(df):
    rows = []
    for model in MODELS:
        for conf_type in CONF_TYPES:
            map_vals = get_series(df, model, conf_type, "map50")
            ece_vals = get_series(df, model, conf_type, "ece")
            rho, p = stats.spearmanr(map_vals, ece_vals)
            rows.append({
                "family": "C_map_vs_ece_spearman",
                "model": model,
                "confidence_type": conf_type,
                "n": int(len(map_vals)),
                "spearman_rho": float(rho),
                "p_value": float(p),
            })
    return bonferroni(rows, n=len(rows))


# ── Group D: clean vs OOD degradation, per model, one-sample vs clean baseline ──

def group_d_clean_vs_ood(df):
    rows = []
    ood_conditions = [c for c in CONDITIONS if c != "clean"]
    for model in MODELS:
        for conf_type in CONF_TYPES:
            for metric in METRICS:
                full = get_series(df, model, conf_type, metric)
                clean_val = full[CONDITIONS.index("clean")]
                ood_vals = np.array([
                    full[CONDITIONS.index(c)] for c in ood_conditions
                ])
                diff = ood_vals - clean_val
                if np.allclose(diff, 0.0):
                    stat, p = 0.0, 1.0
                else:
                    stat, p = stats.wilcoxon(diff)
                rows.append({
                    "family": "D_clean_vs_ood",
                    "model": model,
                    "confidence_type": conf_type,
                    "metric": metric,
                    "n_ood_conditions": int(len(ood_vals)),
                    "clean_value": float(clean_val),
                    "mean_ood_value": float(ood_vals.mean()),
                    "mean_diff_ood_minus_clean": float(diff.mean()),
                    "wilcoxon_stat": float(stat),
                    "p_value": float(p),
                    "cohens_dz": cohens_dz(diff),
                })
    return bonferroni(rows, n=len(rows))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run Group A-D statistical tests over the Stage 5 "
                     "calibration results grid."
    )
    parser.add_argument("--in-csv", default=str(IN_CSV))
    parser.add_argument("--out-json", default=str(OUT_JSON))
    parser.add_argument("--out-csv", default=str(OUT_CSV))
    args = parser.parse_args(argv)

    df = pd.read_csv(args.in_csv)

    all_rows = []
    all_rows += group_a_dcn_vs_ts_vs_raw(df)
    all_rows += group_b_yolov8_vs_rtdetr(df)
    all_rows += group_c_map_vs_ece_spearman(df)
    all_rows += group_d_clean_vs_ood(df)

    with open(args.out_json, "w", encoding="utf-8") as fh:
        json.dump(all_rows, fh, indent=2)

    out_df = pd.DataFrame(all_rows)
    out_df.to_csv(args.out_csv, index=False)

    print(f"wrote {len(all_rows)} test results -> {args.out_json}")
    print(f"wrote {len(all_rows)} test results -> {args.out_csv}")

    for family in ["A_dcn_vs_ts_vs_raw", "B_yolov8_vs_rtdetr",
                    "C_map_vs_ece_spearman", "D_clean_vs_ood"]:
        sub = [r for r in all_rows if r["family"] == family]
        n_sig = sum(1 for r in sub if r["significant_bonferroni_0.05"])
        print(f"\n{family}: {len(sub)} tests, "
              f"{n_sig} significant after Bonferroni (alpha=0.05, n={len(sub)})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
