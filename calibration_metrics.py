from __future__ import annotations

import numpy as np

__all__ = [
    "compute_ece",
    "compute_mce",
    "compute_dece",
    "compute_rmsd",
    "bin_statistics",
]


def _validate(confidences, labels, ious=None):
    conf = np.asarray(confidences, dtype=np.float64).ravel()
    lab = np.asarray(labels, dtype=np.float64).ravel()

    if conf.size != lab.size:
        raise ValueError(
            f"confidences and labels must be the same length "
            f"({conf.size} vs {lab.size})"
        )
    if conf.size == 0:
        raise ValueError("empty input: no predictions to evaluate")
    if np.any(conf < 0.0) or np.any(conf > 1.0):
        raise ValueError("confidences must all lie in [0, 1]")
    if not np.all(np.isin(lab, (0.0, 1.0))):
        raise ValueError("labels must be binary (0 or 1)")

    if ious is None:
        return conf, lab

    iou = np.asarray(ious, dtype=np.float64).ravel()
    if iou.size != conf.size:
        raise ValueError(
            f"ious must be the same length as confidences "
            f"({iou.size} vs {conf.size})"
        )
    if np.any(iou < 0.0) or np.any(iou > 1.0):
        raise ValueError("ious must all lie in [0, 1]")
    return conf, lab, iou


def _bin_indices(values, n_bins):
    idx = np.ceil(values * n_bins).astype(int) - 1
    return np.clip(idx, 0, n_bins - 1)


def bin_statistics(confidences, labels, n_bins=15, min_bin_count=1):
    conf, lab = _validate(confidences, labels)
    total = conf.size

    idx = _bin_indices(conf, n_bins)

    counts = np.bincount(idx, minlength=n_bins).astype(np.float64)
    sum_lab = np.bincount(idx, weights=lab, minlength=n_bins)
    sum_conf = np.bincount(idx, weights=conf, minlength=n_bins)

    keep = counts >= max(1, int(min_bin_count))
    if not np.any(keep):
        raise ValueError(
            f"no bin reached min_bin_count={min_bin_count} "
            f"(n={total} predictions across {n_bins} bins)"
        )

    counts_k = counts[keep]
    acc = sum_lab[keep] / counts_k
    avg_conf = sum_conf[keep] / counts_k

    return {
        "bin_index": np.nonzero(keep)[0],
        "count": counts_k.astype(int),
        "accuracy": acc,
        "confidence": avg_conf,
        "gap": np.abs(acc - avg_conf),
        "weight": counts_k / total,
        "n_total": total,
        "n_bins": n_bins,
        "n_bins_populated": int(counts_k.size),
    }


def compute_ece(confidences, labels, n_bins=15, min_bin_count=1):
    stats = bin_statistics(confidences, labels, n_bins, min_bin_count)
    return float(np.sum(stats["weight"] * stats["gap"]))


def compute_mce(confidences, labels, n_bins=15, min_bin_count=1):
    stats = bin_statistics(confidences, labels, n_bins, min_bin_count)
    return float(np.max(stats["gap"]))


def compute_dece(confidences, labels, ious, n_bins=15,
                 min_bin_count=1, use_netcal=False):
    conf, lab, iou = _validate(confidences, labels, ious)

    if use_netcal:
        try:
            from netcal.metrics import ECE as NetcalECE
        except ImportError as exc:
            raise ImportError(
                "use_netcal=True but netcal is not installed. "
                "Install it (pip install netcal) or leave use_netcal=False "
                "to use the manual 2D binning implementation."
            ) from exc
        X = np.column_stack([conf, iou])
        return float(NetcalECE(bins=n_bins).measure(X, lab.astype(int)))

    total = conf.size
    ci = _bin_indices(conf, n_bins)
    ii = _bin_indices(iou, n_bins)
    flat = ci * n_bins + ii
    ncell = n_bins * n_bins

    counts = np.bincount(flat, minlength=ncell).astype(np.float64)
    sum_lab = np.bincount(flat, weights=lab, minlength=ncell)
    sum_conf = np.bincount(flat, weights=conf, minlength=ncell)

    keep = counts >= max(1, int(min_bin_count))
    if not np.any(keep):
        raise ValueError(
            f"no confidence x IoU cell reached min_bin_count={min_bin_count} "
            f"(n={total} predictions across {ncell} cells)"
        )

    c = counts[keep]
    acc = sum_lab[keep] / c
    avg_conf = sum_conf[keep] / c
    return float(np.sum((c / total) * np.abs(acc - avg_conf)))


def dece_diagnostics(confidences, labels, ious, n_bins=15):
    conf, lab, iou = _validate(confidences, labels, ious)
    flat = _bin_indices(conf, n_bins) * n_bins + _bin_indices(iou, n_bins)
    counts = np.bincount(flat, minlength=n_bins * n_bins)
    nz = counts[counts > 0]
    return {
        "n_predictions": int(conf.size),
        "n_cells": int(n_bins * n_bins),
        "n_cells_populated": int(nz.size),
        "fraction_cells_populated": float(nz.size / (n_bins * n_bins)),
        "median_cell_count": float(np.median(nz)) if nz.size else 0.0,
        "cells_with_one_sample": int(np.sum(nz == 1)),
    }


def compute_rmsd(confidences, labels, n_bins=15, min_bin_count=1,
                 weighted=True):
    stats = bin_statistics(confidences, labels, n_bins, min_bin_count)
    sq = stats["gap"] ** 2
    if weighted:
        w = stats["weight"]
        return float(np.sqrt(np.sum(w * sq) / np.sum(w)))
    return float(np.sqrt(np.mean(sq)))


def _make_wellcalibrated(n, rng):
    conf = rng.uniform(0.0, 1.0, size=n)
    lab = (rng.uniform(size=n) < conf).astype(int)
    return conf, lab


def _make_overconfident(n, rng):
    conf = rng.uniform(0.6, 1.0, size=n)
    true_p = conf * 0.5
    lab = (rng.uniform(size=n) < true_p).astype(int)
    return conf, lab


def _make_ious(conf, lab, rng):
    iou = np.where(
        lab == 1,
        rng.uniform(0.5, 0.95, size=conf.size),
        rng.uniform(0.0, 0.45, size=conf.size),
    )
    return np.clip(iou, 0.0, 1.0)


def _run_tests():
    rng = np.random.default_rng(0)
    n = 5000

    print("=" * 66)
    print("calibration_metrics.py -- synthetic sanity checks")
    print("=" * 66)

    for name, maker in (
        ("well-calibrated", _make_wellcalibrated),
        ("overconfident", _make_overconfident),
    ):
        conf, lab = maker(n, rng)
        iou = _make_ious(conf, lab, rng)

        ece = compute_ece(conf, lab)
        mce = compute_mce(conf, lab)
        dece = compute_dece(conf, lab, iou)
        rmsd = compute_rmsd(conf, lab)

        print(f"\n[{name}]  n={n}, hit rate={lab.mean():.3f}")
        print(f"  ECE   = {ece:.4f}")
        print(f"  MCE   = {mce:.4f}")
        print(f"  D-ECE = {dece:.4f}   (manual 2D binning)")
        print(f"  RMSD  = {rmsd:.4f}   (root mean square, weighted)")

        assert 0.0 <= ece <= 1.0
        assert 0.0 <= mce <= 1.0
        assert 0.0 <= dece <= 1.0
        assert 0.0 <= rmsd <= 1.0
        assert mce >= ece - 1e-9, "MCE must be >= ECE by construction"

    c_ok, l_ok = _make_wellcalibrated(n, rng)
    c_bad, l_bad = _make_overconfident(n, rng)
    ece_ok, ece_bad = compute_ece(c_ok, l_ok), compute_ece(c_bad, l_bad)
    assert ece_bad > ece_ok, (ece_ok, ece_bad)
    print(f"\n[direction] ECE well-calibrated {ece_ok:.4f} "
          f"< overconfident {ece_bad:.4f}  OK")

    print("\n[edge cases]")
    perfect_c = np.array([0.0, 0.0, 1.0, 1.0])
    perfect_l = np.array([0, 0, 1, 1])
    print(f"  perfectly separated ECE = {compute_ece(perfect_c, perfect_l):.4f}"
          "   (expect 0.0000)")

    single = compute_ece([0.7], [1])
    print(f"  single prediction  ECE = {single:.4f}   (expect 0.3000)")

    for bad_call, why in (
        (lambda: compute_ece([0.5, 1.2], [1, 0]), "confidence > 1"),
        (lambda: compute_ece([0.5, 0.6], [1, 2]), "non-binary label"),
        (lambda: compute_ece([0.5], [1, 0]), "length mismatch"),
        (lambda: compute_ece([], []), "empty input"),
    ):
        try:
            bad_call()
        except ValueError:
            print(f"  rejected: {why}  OK")
        else:
            raise AssertionError(f"should have rejected: {why}")

    conf_real = np.clip(rng.beta(8, 2, size=2000), 0, 1)
    lab_real = (rng.uniform(size=2000) < conf_real * 0.8).astype(int)
    iou_real = _make_ious(conf_real, lab_real, rng)
    diag = dece_diagnostics(conf_real, lab_real, iou_real, n_bins=15)
    print("\n[D-ECE grid sparsity at n_bins=15, detection-shaped data]")
    for k, v in diag.items():
        print(f"  {k}: {v}")
    print("  -> if fraction_cells_populated is low, report D-ECE at a")
    print("     coarser n_bins and say so in your methodology.")

    print("\n" + "=" * 66)
    print("all checks passed")
    print("=" * 66)


if __name__ == "__main__":
    _run_tests()