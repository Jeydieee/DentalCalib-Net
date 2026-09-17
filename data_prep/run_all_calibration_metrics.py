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
from compute_map import greedy_ap, load_gt_boxes_by_filename


ROOT_DIR = Path(__file__).resolve().parent.parent

LABELED_DIR = ROOT_DIR / "stage4_outputs/labeled_predictions"
RECAL_DIR = ROOT_DIR / "stage4_outputs/recalibrated_predictions"

MODELS = ["yolov8", "rtdetr"]

CORRUPTION_TYPES = [
    "brightness_variation", "gaussian_noise", "jpeg_compression", "motion_blur"
]
SEVERITIES = [1, 2, 3, 4, 5]
CLEAN_CONDITION = "clean"

CONDITION_PATTERN = "{corruption}_S{severity}"

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
GT_JSON = ROOT_DIR / "data_prep/repool_output/dentex_merged_1005_resized.json"

OUT_JSON = ROOT_DIR / "stage5_outputs/calibration_results.json"
OUT_CSV = ROOT_DIR / "stage5_outputs/calibration_results.csv"

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


def labeled_path_for(labeled_dir, model, condition, corruption=None, severity=None):
    labeled_dir = Path(labeled_dir)
    if model == "rtdetr":
        if condition == CLEAN_CONDITION:
            return labeled_dir / "rtdetr_predictions_clean_filtered_labeled.json"
        return labeled_dir / f"rtdetr_{corruption}_S{severity}_filtered_labeled.json"
    if condition == CLEAN_CONDITION:
        return labeled_dir / "yolov8_predictions_clean_labeled.json"
    return labeled_dir / f"yolov8_{corruption}_S{severity}_labeled.json"


def recal_path_for(recal_dir, model, condition, corruption=None, severity=None):
    recal_dir = Path(recal_dir)
    if condition == CLEAN_CONDITION:
        return recal_dir / f"{model}_clean_recalibrated.json"
    return recal_dir / f"{model}_{corruption}_S{severity}_recalibrated.json"


def _load_json_list(path):
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    if isinstance(raw, dict):
        for key in ("images", "results", "data", "entries"):
            if isinstance(raw.get(key), list):
                raw = raw[key]
                break
        else:
            raise ValueError(
                f"{path}: top-level object has no list under images/"
                f"results/data/entries; keys present: {sorted(raw.keys())}"
            )
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a list of image entries")
    return raw


def load_file(labeled_path, recal_path):
    labeled = _load_json_list(labeled_path)
    recal = _load_json_list(recal_path)

    recal_by_image = {e["image_path"]: e["predictions"] for e in recal}

    conf = {k: [] for k in CONFIDENCE_TYPES.values()}
    labels, ious, filenames, bboxes = [], [], [], []
    n_gt = 0
    saw_gt = False
    skipped = 0

    for entry in labeled:
        for key in ("num_ground_truth", "n_ground_truth", "num_gt", "n_gt"):
            if key in entry:
                n_gt += int(entry[key])
                saw_gt = True
                break
        else:
            if isinstance(entry.get("ground_truth"), list):
                n_gt += len(entry["ground_truth"])
                saw_gt = True

        image_path = entry["image_path"]
        recal_preds = recal_by_image.get(image_path)
        if recal_preds is None:
            raise ValueError(
                f"{recal_path}: no entry for image_path {image_path!r} "
                f"found in {labeled_path}"
            )
        lab_preds = entry.get("predictions", []) or []
        if len(lab_preds) != len(recal_preds):
            raise ValueError(
                f"prediction count mismatch for {image_path!r}: "
                f"{len(lab_preds)} in {labeled_path} vs "
                f"{len(recal_preds)} in {recal_path}. Labeled and "
                f"recalibrated files must come from the same source "
                f"predictions in the same order -- do not merge by index "
                f"across mismatched files."
            )

        filename = Path(image_path).name
        for lab_pred, recal_pred in zip(lab_preds, recal_preds):
            if LABEL_KEY not in lab_pred or IOU_KEY not in lab_pred:
                skipped += 1
                continue
            if any(k not in recal_pred for k in conf):
                skipped += 1
                continue
            for k in conf:
                conf[k].append(float(recal_pred[k]))
            labels.append(int(lab_pred[LABEL_KEY]))
            ious.append(float(lab_pred[IOU_KEY]))
            filenames.append(filename)
            bboxes.append(recal_pred.get("bbox_xyxy") or lab_pred.get("bbox_xyxy"))

    if not labels:
        raise ValueError("no usable predictions found in file")

    labels_arr = np.asarray(labels, dtype=np.int64)
    if labels_arr.sum() == 0 and labels_arr.size > 20:
        print(f"  NOTE: {labeled_path.name} -- every one of {labels_arr.size} "
              f"predictions has label=0 (best_iou never reached the match "
              f"threshold). For RT-DETR gaussian_noise S2-S5 this is the "
              f"confirmed model failure (see project notes), not a data "
              f"bug -- metrics are still computed and are meaningful "
              f"(near-maximal ECE reflects total miscalibration). For any "
              f"other condition, treat this as suspicious and verify.")

    return {
        "conf": {k: np.asarray(v, dtype=np.float64) for k, v in conf.items()},
        "labels": np.asarray(labels, dtype=np.int64),
        "ious": np.asarray(ious, dtype=np.float64),
        "filenames": filenames,
        "bboxes": bboxes,
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


def score_file(data, n_gt, gt_boxes_by_filename):
    labels = data["labels"]
    ious = data["ious"]
    filenames = data["filenames"]
    bboxes = data["bboxes"]
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

        ap = None
        if n_gt and gt_boxes_by_filename:
            dets = [
                {"filename": fn, "bbox": bb, "confidence": float(c)}
                for fn, bb, c in zip(filenames, bboxes, conf)
            ]
            try:
                ap = greedy_ap(dets, gt_boxes_by_filename, n_gt,
                               method=MAP_METHOD)["ap"]
            except Exception as exc:
                notes.append(
                    f"{short}: mAP computation failed "
                    f"({type(exc).__name__}: {exc}); ECE/MCE/D-ECE/RMSD "
                    f"above are still valid."
                )

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
    parser.add_argument("--labeled-dir", default=str(LABELED_DIR))
    parser.add_argument("--recal-dir", default=str(RECAL_DIR))
    parser.add_argument("--gt-json", default=str(GT_JSON),
                        help="ground-truth annotations JSON, used for "
                             "greedy one-detection-per-box AP matching")
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
    print(f"Labeled dir:      {args.labeled_dir}")
    print(f"Recalibrated dir: {args.recal_dir}")
    print("clean condition source: stage4 test-split clean files "
          "(labeled + recalibrated)")
    print()

    if args.dry_run:
        found = missing = 0
        for model, cond, corr, sev in expected:
            lp = labeled_path_for(args.labeled_dir, model, cond, corr, sev)
            rp = recal_path_for(args.recal_dir, model, cond, corr, sev)
            ok = lp.exists() and rp.exists()
            found += ok
            missing += not ok
            status = "OK     " if ok else "MISSING"
            print(f"  {status}  {lp}")
            print(f"  {status}  {rp}")
        print(f"\n{found} found, {missing} missing")
        return 0 if missing == 0 else 1

    gt_boxes_by_filename = load_gt_boxes_by_filename(args.gt_json)
    print(f"Ground truth (for greedy AP matching): {args.gt_json}  "
          f"({sum(len(v) for v in gt_boxes_by_filename.values())} boxes "
          f"across {len(gt_boxes_by_filename)} images)\n")

    results = []
    failures = []
    gt_seen = set()

    for model, cond, corr, sev in expected:
        lp = labeled_path_for(args.labeled_dir, model, cond, corr, sev)
        rp = recal_path_for(args.recal_dir, model, cond, corr, sev)
        path = f"{lp} + {rp}"
        try:
            data = load_file(lp, rp)
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
            rows, notes = score_file(data, n_gt, gt_boxes_by_filename)
        except Exception as exc:
            failures.append((str(path), f"metric failure: "
                                        f"{type(exc).__name__}: {exc}"))
            traceback.print_exc(file=sys.stderr)
            continue

        for note in notes:
            print(f"  note: {lp.name} -- {note}")

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