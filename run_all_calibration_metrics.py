from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from pathlib import Path

import numpy as np

from calibration_metrics import (
    compute_ece,
    compute_mce,
    compute_dece,
    compute_rmsd,
    bin_statistics,
)
from compute_map import compute_ap


PRED_DIR = Path("stage2_outputs")

MODELS = ["yolov8", "rtdetr"]

CORRUPTION_TYPES = [
    "brightness_variation", "gaussian_noise", "jpeg_compression", "motion_blur"
]
SEVERITIES = [1, 2, 3, 4, 5]
CLEAN_CONDITION = "clean"

CONDITION_PATTERN = "{corruption}_S{severity}"

USE_VAL_AS_CLEAN = False

CONFIDENCE_TYPES = {
    "raw": "confidence_raw",
    "ts": "confidence_ts",
    "dcn": "confidence_dcn",
}

LABEL_KEY = "label"
IOU_KEY = "best_iou"

N_BINS = 15
DECE_N_BINS = 15
MIN_BIN_COUNT = 1
MAP_METHOD = "all_point"

GT_COUNT = 175

OUT_JSON = Path("calibration_results.json")
OUT_CSV = Path("calibration_results.csv")

COLUMNS = [
    "model", "condition", "corruption_type", "severity", "confidence_type",
    "ece", "mce", "dece", "rmsd", "map50",
    "n_predictions", "n_bins_populated", "map_is_tied_to_raw",
]


def build_conditions():
    out = [(CLEAN_CONDITION, "none", 0)]
    for corr in CORRUPTION_TYPES:
        for sev in SEVERITIES:
            out.append((
                CONDITION_PATTERN.format(corruption=corr, severity=sev),
                corr,
                sev,
            ))
    return out


def path_for(pred_dir, model, condition, corruption=None, severity=None):
    pred_dir = Path(pred_dir)
    if condition == CLEAN_CONDITION:
        return pred_dir / f"{model}_predictions_clean.json"
    return (pred_dir / f"{model}_ood_predictions"
            / f"{model}_{corruption}_S{severity}.json")


def load_file(path):
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    if isinstance(raw, dict):
        for key in ("images", "results", "data", "entries"):
            if isinstance(raw.get(key), list):
                raw = raw[key]
                break
        else:
            raise ValueError(
                f"top-level object has no list under images/results/data/"
                f"entries; keys present: {sorted(raw.keys())}"
            )
    if not isinstance(raw, list):
        raise ValueError("expected a list of image entries")

    conf = {k: [] for k in CONFIDENCE_TYPES.values()}
    labels, ious = [], []
    n_gt = 0
    saw_gt = False
    skipped = 0

    for entry in raw:
        for key in ("num_ground_truth", "n_ground_truth", "num_gt", "n_gt"):
            if key in entry:
                n_gt += int(entry[key])
                saw_gt = True
                break
        else:
            if isinstance(entry.get("ground_truth"), list):
                n_gt += len(entry["ground_truth"])
                saw_gt = True

        for pred in entry.get("predictions", []) or []:
            if LABEL_KEY not in pred or IOU_KEY not in pred:
                skipped += 1
                continue
            if any(k not in pred for k in conf):
                skipped += 1
                continue
            for k in conf:
                conf[k].append(float(pred[k]))
            labels.append(int(pred[LABEL_KEY]))
            ious.append(float(pred[IOU_KEY]))

    if not labels:
        raise ValueError("no usable predictions found in file")

    labels_arr = np.asarray(labels, dtype=np.int64)
    if labels_arr.sum() == 0 and labels_arr.size > 20:
        raise ValueError(
            f"every one of {labels_arr.size} predictions in this file has "
            f"label=0 (best_iou never reached the match threshold). This is "
            f"the exact symptom of the known filename mismatch between "
            f"prediction image_path values (test_N.png-style) and "
            f"dentex_merged_1005_resized.json (original DENTEX filenames) -- "
            f"see the GT_COUNT comment at the top of this file. Refusing to "
            f"report metrics computed from an all-zero label file; fix the "
            f"filename mapping in batch_iou_labeling.py first."
        )

    return {
        "conf": {k: np.asarray(v, dtype=np.float64) for k, v in conf.items()},
        "labels": np.asarray(labels, dtype=np.int64),
        "ious": np.asarray(ious, dtype=np.float64),
        "n_gt": n_gt if saw_gt else None,
        "skipped": skipped,
    }


def resolve_gt(data, args):
    if args.n_gt is not None:
        return int(args.n_gt)
    if GT_COUNT is not None:
        return int(GT_COUNT)
    if data["n_gt"]:
        return int(data["n_gt"])
    return None


def score_file(data, n_gt):
    labels = data["labels"]
    ious = data["ious"]
    rows, notes = [], []

    raw_order = None

    for short, key in CONFIDENCE_TYPES.items():
        conf = data["conf"][key]

        ece = compute_ece(conf, labels, N_BINS, MIN_BIN_COUNT)
        mce = compute_mce(conf, labels, N_BINS, MIN_BIN_COUNT)
        dece = compute_dece(conf, labels, ious, DECE_N_BINS, MIN_BIN_COUNT)
        rmsd = compute_rmsd(conf, labels, N_BINS, MIN_BIN_COUNT)
        stats = bin_statistics(conf, labels, N_BINS, MIN_BIN_COUNT)

        order = np.argsort(-conf, kind="stable")
        tied = False
        if short == "raw":
            raw_order = order
        elif raw_order is not None:
            tied = bool(np.array_equal(order, raw_order))

        if n_gt:
            ap = compute_ap(labels[order], n_gt, method=MAP_METHOD)["ap"]
        else:
            ap = None

        rows.append({
            "confidence_type": short,
            "ece": ece,
            "mce": mce,
            "dece": dece,
            "rmsd": rmsd,
            "map50": ap,
            "n_predictions": int(labels.size),
            "n_bins_populated": stats["n_bins_populated"],
            "map_is_tied_to_raw": tied,
        })

    return rows, notes


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run all calibration metrics across the Stage 5 grid."
    )
    parser.add_argument("--pred-dir", default=str(PRED_DIR))
    parser.add_argument("--use-val-as-clean", action="store_true",
                        default=USE_VAL_AS_CLEAN,
                        help="use {model}_predictions_val.json as the "
                             "'clean' condition instead of "
                             "{model}_predictions_clean.json. Do not set "
                             "this until you've confirmed which split "
                             "belongs in the reported grid -- see the "
                             "USE_VAL_AS_CLEAN comment at the top of this "
                             "file.")
    parser.add_argument("--n-gt", type=int, default=None,
                        help="ground-truth object count (same for all "
                             "conditions; corruptions do not change the GT)")
    parser.add_argument("--dry-run", action="store_true",
                        help="list expected paths and report missing ones, "
                             "then exit without computing anything")
    parser.add_argument("--out-json", default=str(OUT_JSON))
    parser.add_argument("--out-csv", default=str(OUT_CSV))
    args = parser.parse_args(argv)

    conditions = build_conditions()
    expected = [
        (m, c, corr, sev)
        for m in MODELS
        for (c, corr, sev) in conditions
    ]

    print(f"Grid: {len(MODELS)} models x {len(conditions)} conditions "
          f"= {len(expected)} files -> "
          f"{len(expected) * len(CONFIDENCE_TYPES)} rows")
    print(f"Prediction dir: {args.pred_dir}")
    if args.use_val_as_clean:
        print("clean condition source: {model}_predictions_VAL.json  "
              "(--use-val-as-clean was set)")
    else:
        print("clean condition source: {model}_predictions_clean.json  "
              "(default)")
        print("  NOTE: this repo also has {model}_predictions_val.json, "
              "separate\n  from _clean.json. If val is actually what should "
              "be reported as\n  the 'clean' condition, rerun with "
              "--use-val-as-clean. Do not\n  guess -- confirm which split "
              "Table 9 is supposed to reflect.")
    print()

    def resolve_clean_path(pred_dir, model):
        suffix = "val" if args.use_val_as_clean else "clean"
        return Path(pred_dir) / f"{model}_predictions_{suffix}.json"

    def resolve_path(pred_dir, model, cond, corr, sev):
        if cond == CLEAN_CONDITION:
            return resolve_clean_path(pred_dir, model)
        return path_for(pred_dir, model, cond, corr, sev)

    if args.dry_run:
        found = missing = 0
        for model, cond, corr, sev in expected:
            p = resolve_path(args.pred_dir, model, cond, corr, sev)
            ok = p.exists()
            found += ok
            missing += not ok
            print(f"  {'OK     ' if ok else 'MISSING'}  {p}")
        print(f"\n{found} found, {missing} missing")
        if missing:
            print("\nIf files are missing, check the layout above against "
                  "your\nactual repo. Per the delegation doc, large files "
                  "excluded from\ngit may need downloading from the team "
                  "Drive first.")
        return 0 if missing == 0 else 1

    results = []
    failures = []
    gt_seen = set()

    for model, cond, corr, sev in expected:
        path = resolve_path(args.pred_dir, model, cond, corr, sev)
        try:
            data = load_file(path)
        except FileNotFoundError:
            failures.append((str(path), "file not found"))
            continue
        except json.JSONDecodeError as exc:
            failures.append((str(path), f"invalid JSON: {exc}"))
            continue
        except Exception as exc:
            failures.append((str(path), f"{type(exc).__name__}: {exc}"))
            continue

        n_gt = resolve_gt(data, args)
        if n_gt:
            gt_seen.add(n_gt)

        try:
            rows, _ = score_file(data, n_gt)
        except Exception as exc:
            failures.append((str(path), f"metric failure: "
                                        f"{type(exc).__name__}: {exc}"))
            traceback.print_exc(file=sys.stderr)
            continue

        if data["skipped"]:
            print(f"  note: {path.name} -- skipped {data['skipped']} "
                  f"prediction(s) missing required keys")

        for r in rows:
            results.append({
                "model": model,
                "condition": cond,
                "corruption_type": corr,
                "severity": sev,
                **r,
            })

    if results:
        with open(args.out_json, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        with open(args.out_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=COLUMNS)
            w.writeheader()
            for r in results:
                w.writerow({k: r.get(k) for k in COLUMNS})

    expected_rows = len(expected) * len(CONFIDENCE_TYPES)
    print("\n" + "=" * 66)
    print(f"rows computed : {len(results)} / {expected_rows}")
    print(f"files failed  : {len(failures)} / {len(expected)}")

    if failures:
        print("\nFAILURES:")
        for p, why in failures:
            print(f"  {p}\n    {why}")

    if results:
        print(f"\nwrote {args.out_json}")
        print(f"wrote {args.out_csv}")

        n_no_map = sum(1 for r in results if r["map50"] is None)
        if n_no_map:
            print(f"\nWARNING: map50 is null on {n_no_map} row(s) -- no "
                  f"ground-truth\ncount was available. Groups B4/B5 (Spearman "
                  f"mAP vs ECE) and\nTables 9-16 cannot be completed from "
                  f"this file. Pass --n-gt.")

        n_tied = sum(1 for r in results if r["map_is_tied_to_raw"])
        if n_tied:
            print(f"\nNOTE: {n_tied} row(s) have map50 identical to their "
                  f"raw counterpart.")
            print("This is expected for Temperature Scaling: TS is a "
                  "monotonic\ntransform, so it cannot change the confidence "
                  "ranking, so it\ncannot change mAP. Only calibration "
                  "metrics move.")
            print("Consequences to handle deliberately, not discover later:")
            print("  - B4/B5 correlate mAP against ECE across severities. "
                  "With TS,\n    mAP is the raw curve; only ECE differs. Say "
                  "so explicitly.")
            print("  - Tables 14-16 will show an unchanging mAP column "
                  "across\n    raw and TS. That is correct, not a bug. "
                  "Footnote it.")
            print("  - If DCN rows are ALSO tied, DCN is monotonic too and "
                  "is not\n    reordering detections either. If DCN rows are "
                  "NOT tied, DCN\n    changes the ranking and its mAP is "
                  "genuinely different.")

        dcn_tied = [r for r in results
                    if r["confidence_type"] == "dcn" and r["map_is_tied_to_raw"]]
        print(f"\nDCN rows tied to raw: {len(dcn_tied)} / "
              f"{len(expected)} -- "
              f"{'DCN is monotonic' if dcn_tied else 'DCN reorders detections'}")

        if len(gt_seen) > 1:
            print(f"\nWARNING: more than one ground-truth count was used "
                  f"across\nconditions: {sorted(gt_seen)}. Corruptions do not "
                  f"change the number\nof teeth in an image, so these should "
                  f"all be equal. Investigate.")

        sparse = [r for r in results if r["n_bins_populated"] < N_BINS // 2]
        if sparse:
            print(f"\nWARNING: {len(sparse)} row(s) populated fewer than "
                  f"{N_BINS // 2} of\n{N_BINS} confidence bins. MCE and RMSD "
                  f"on those rows rest on very\nfew samples. Check before "
                  f"putting them in a table.")

    print("=" * 66)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())