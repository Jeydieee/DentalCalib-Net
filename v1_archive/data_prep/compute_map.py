from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict

import numpy as np

__all__ = [
    "compute_ap",
    "compute_map",
    "load_predictions",
    "load_gt_boxes_by_filename",
    "greedy_match",
    "greedy_ap",
]


def _box_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def load_gt_boxes_by_filename(gt_json_path, category_id_3=0):
    with open(gt_json_path, "r", encoding="utf-8") as fh:
        merged = json.load(fh)

    images_by_id = {img["id"]: img["file_name"] for img in merged["images"]}

    gt_by_filename = defaultdict(list)
    for ann in merged["annotations"]:
        if ann.get("category_id_3") != category_id_3:
            continue
        file_name = images_by_id.get(ann["image_id"])
        if file_name is None:
            continue
        x, y, w, h = ann["bbox"]
        gt_by_filename[file_name].append([x, y, x + w, y + h])

    return dict(gt_by_filename)


def greedy_match(dets, gt_boxes_by_filename, iou_threshold=0.5):
    """Standard COCO/VOC-style greedy matching: each detection is scored in

    descending confidence order, and can only match a ground-truth box that
    no earlier (higher-confidence) detection has already claimed. This is
    required so tp never exceeds the ground-truth count -- unlike the raw
    per-prediction 'label' field (IoU >= threshold to ANY GT box), which
    allows multiple overlapping detections to all be counted correct against
    the same GT box.

    dets: list of {"filename": str, "bbox": [x1,y1,x2,y2], "confidence": float}
          already in the desired scoring order is NOT required -- this
          function sorts by confidence itself.

    Returns (labels, order): labels[i] is 1/0 for dets[i] (original order),
    order is the confidence-descending index order used for AP.
    """
    n = len(dets)
    order = sorted(range(n), key=lambda i: -dets[i]["confidence"])

    claimed = {
        fname: [False] * len(boxes)
        for fname, boxes in gt_boxes_by_filename.items()
    }

    labels = np.zeros(n, dtype=np.int64)
    for i in order:
        d = dets[i]
        gt_boxes = gt_boxes_by_filename.get(d["filename"], [])
        used = claimed.get(d["filename"])
        best_iou, best_j = 0.0, -1
        for j, gt_box in enumerate(gt_boxes):
            if used is not None and used[j]:
                continue
            iou = _box_iou(d["bbox"], gt_box)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_threshold and best_j >= 0:
            labels[i] = 1
            claimed[d["filename"]][best_j] = True

    return labels, np.asarray(order, dtype=np.int64)


def greedy_ap(dets, gt_boxes_by_filename, n_ground_truth, iou_threshold=0.5,
              method="all_point"):
    labels, order = greedy_match(dets, gt_boxes_by_filename, iou_threshold)
    return compute_ap(labels[order], n_ground_truth, method=method)


def compute_ap(labels_sorted, n_ground_truth, method="all_point"):
    if n_ground_truth <= 0:
        raise ValueError(
            "n_ground_truth must be > 0; recall is undefined otherwise"
        )

    lab = np.asarray(labels_sorted, dtype=np.float64).ravel()
    if lab.size == 0:
        return {
            "ap": 0.0,
            "precision": np.array([]),
            "recall": np.array([]),
            "tp": 0,
            "fp": 0,
            "n_ground_truth": int(n_ground_truth),
        }
    if not np.all(np.isin(lab, (0.0, 1.0))):
        raise ValueError("labels must be binary (0 or 1)")

    tp_cum = np.cumsum(lab)
    fp_cum = np.cumsum(1.0 - lab)

    recall = tp_cum / float(n_ground_truth)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, np.finfo(np.float64).eps)

    if tp_cum[-1] > n_ground_truth:
        raise ValueError(
            f"found {int(tp_cum[-1])} true positives but only "
            f"{n_ground_truth} ground-truth objects. Either the ground-truth "
            f"count is wrong, or Stage 4 allowed duplicate detections to be "
            f"labelled as matches. Fix this before reporting AP."
        )

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    if method == "all_point":
        idx = np.nonzero(mrec[1:] != mrec[:-1])[0]
        ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    elif method in ("coco_101", "voc_11"):
        points = (np.linspace(0.0, 1.0, 101) if method == "coco_101"
                  else np.linspace(0.0, 1.0, 11))
        ap = float(np.mean(np.interp(points, mrec, mpre, left=mpre[0],
                                     right=0.0)))
    else:
        raise ValueError(f"unknown method: {method!r}")

    return {
        "ap": ap,
        "precision": precision,
        "recall": recall,
        "tp": int(tp_cum[-1]),
        "fp": int(fp_cum[-1]),
        "n_ground_truth": int(n_ground_truth),
        "max_recall": float(recall[-1]),
    }


def load_predictions(path):
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    if isinstance(raw, dict):
        for key in ("images", "results", "data", "entries"):
            if key in raw and isinstance(raw[key], list):
                raw = raw[key]
                break
        else:
            raise ValueError(
                f"{path}: top-level object has no list under 'images', "
                f"'results', 'data' or 'entries'. Keys present: "
                f"{sorted(raw.keys())}"
            )
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a list of image entries")

    rows = []
    gt_per_image = {}
    missing_conf = 0
    missing_label = 0

    for entry in raw:
        img = entry.get("image_path", entry.get("image_id", "<unknown>"))

        for key in ("num_ground_truth", "n_ground_truth", "num_gt", "n_gt"):
            if key in entry:
                gt_per_image[img] = int(entry[key])
                break
        else:
            if isinstance(entry.get("ground_truth"), list):
                gt_per_image[img] = len(entry["ground_truth"])

        for pred in entry.get("predictions", []) or []:
            if "confidence" not in pred:
                missing_conf += 1
                continue
            if "label" not in pred:
                missing_label += 1
                continue
            cls = pred.get("class_id", pred.get("category_id", 0))
            rows.append({
                "image": img,
                "class_id": cls,
                "confidence": float(pred["confidence"]),
                "label": int(pred["label"]),
                "bbox": pred.get("bbox_xyxy"),
            })

    if missing_conf or missing_label:
        print(f"  WARNING: skipped {missing_conf} prediction(s) with no "
              f"'confidence' and {missing_label} with no 'label'",
              file=sys.stderr)

    return rows, gt_per_image


def _resolve_gt_counts(rows, gt_per_image, args):
    classes = sorted({r["class_id"] for r in rows})

    if args.gt_json:
        with open(args.gt_json, "r", encoding="utf-8") as fh:
            gt_raw = json.load(fh)
        counts = defaultdict(int)
        entries = gt_raw if isinstance(gt_raw, list) else gt_raw.get(
            "annotations", gt_raw.get("images", []))
        for entry in entries:
            if "annotations" in entry or "ground_truth" in entry:
                boxes = entry.get("annotations", entry.get("ground_truth", []))
                for box in boxes:
                    counts[box.get("class_id",
                                   box.get("category_id", 0))] += 1
            else:
                counts[entry.get("class_id", entry.get("category_id", 0))] += 1
        if not counts:
            raise ValueError(f"{args.gt_json}: found no ground-truth boxes")
        print(f"  ground truth from {args.gt_json}: "
              f"{sum(counts.values())} boxes across {len(counts)} class(es)")
        return dict(counts)

    if args.n_gt is not None:
        if len(classes) > 1:
            raise ValueError(
                f"--n-gt is a single number but the predictions contain "
                f"{len(classes)} classes ({classes}). Use --gt-json so each "
                f"class gets its own ground-truth count."
            )
        print(f"  ground truth from --n-gt: {args.n_gt}")
        return {classes[0]: int(args.n_gt)}

    if gt_per_image:
        covered = {r["image"] for r in rows}
        missing = covered - set(gt_per_image)
        if missing:
            raise ValueError(
                f"{len(missing)} image(s) have predictions but no "
                f"ground-truth count (e.g. {sorted(missing)[:3]}). Partial "
                f"counts would silently understate recall."
            )
        total = sum(gt_per_image.values())
        if len(classes) > 1:
            raise ValueError(
                "per-image ground-truth counts are not broken down by class, "
                "but the predictions have multiple classes. Use --gt-json."
            )
        print(f"  ground truth from per-image keys: {total} boxes across "
              f"{len(gt_per_image)} image(s)")
        return {classes[0]: total}

    if args.assume_no_misses:
        counts = defaultdict(int)
        for r in rows:
            if r["label"] == 1:
                counts[r["class_id"]] += 1
        print("", file=sys.stderr)
        print("  " + "!" * 62, file=sys.stderr)
        print("  !! --assume-no-misses IS ON.", file=sys.stderr)
        print("  !! Ground truth is being set equal to the true-positive "
              "count,", file=sys.stderr)
        print("  !! which forces recall to 1.0 and INFLATES AP.",
              file=sys.stderr)
        print("  !! This number is not valid for the thesis. Diagnostic "
              "use only.", file=sys.stderr)
        print("  " + "!" * 62, file=sys.stderr)
        print("", file=sys.stderr)
        if not counts:
            raise ValueError("no true positives; cannot even fake a GT count")
        return dict(counts)

    raise SystemExit(
        "\nERROR: no ground-truth count available.\n\n"
        "mAP needs the total number of ground-truth objects to compute\n"
        "recall. Your predictions file lists only predictions, so missed\n"
        "detections (false negatives) are invisible to this script.\n\n"
        "Supply one of:\n"
        "  --gt-json PATH   annotations file with the ground-truth boxes\n"
        "  --n-gt N         total ground-truth box count (single class only)\n"
        "  add a 'num_ground_truth' key to each image entry in the\n"
        "  predictions JSON\n\n"
        "Refusing to guess, because guessing produces a plausible-looking\n"
        "but wrong number.\n"
    )


def compute_map(rows, gt_counts, method="all_point", verbose=True):
    by_class = defaultdict(list)
    for r in rows:
        by_class[r["class_id"]].append(r)

    per_class = {}
    for cls in sorted(set(by_class) | set(gt_counts)):
        preds = by_class.get(cls, [])
        preds.sort(key=lambda r: r["confidence"], reverse=True)
        n_gt = gt_counts.get(cls, 0)
        if n_gt <= 0:
            if verbose:
                print(f"  skipping class {cls}: no ground-truth objects")
            continue
        per_class[cls] = compute_ap(
            [r["label"] for r in preds], n_gt, method=method
        )

    if not per_class:
        raise ValueError("no class had any ground-truth objects")

    map_value = float(np.mean([v["ap"] for v in per_class.values()]))
    return map_value, per_class


def _check_duplicates(rows):
    tps = defaultdict(int)
    for r in rows:
        if r["label"] == 1:
            tps[r["image"]] += 1
    if not tps:
        return
    counts = np.array(list(tps.values()))
    print(f"  true positives per image: min={counts.min()} "
          f"median={int(np.median(counts))} max={counts.max()}")
    print("  -> if max is far above the number of ground-truth objects you "
          "expect\n     in one image, Stage 4 may be labelling duplicate "
          "detections as\n     matches. Check before trusting AP.")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compute mAP@0.50 from a predictions JSON."
    )
    parser.add_argument("predictions", help="path to the predictions JSON")
    parser.add_argument("--gt-json", default=None,
                        help="ground-truth annotations JSON (preferred)")
    parser.add_argument("--n-gt", type=int, default=None,
                        help="total ground-truth box count, single class only")
    parser.add_argument("--assume-no-misses", action="store_true",
                        help="DIAGNOSTIC ONLY: set GT = true-positive count. "
                             "Inflates AP. Never use for reported results.")
    parser.add_argument("--method", default="all_point",
                        choices=["all_point", "coco_101", "voc_11"],
                        help="interpolation scheme (default: all_point)")
    parser.add_argument("--check-duplicates", action="store_true",
                        help="report per-image true-positive counts")
    parser.add_argument("--json-out", default=None,
                        help="write the full result to this path")
    args = parser.parse_args(argv)

    print(f"Loading {args.predictions}")
    rows, gt_per_image = load_predictions(args.predictions)
    print(f"  {len(rows)} prediction(s) across "
          f"{len({r['image'] for r in rows})} image(s)")

    if args.check_duplicates:
        _check_duplicates(rows)

    gt_counts = _resolve_gt_counts(rows, gt_per_image, args)

    map_value, per_class = compute_map(rows, gt_counts, method=args.method)

    single = len(per_class) == 1
    name = "AP@0.50" if single else "mAP@0.50"

    print(f"\n  interpolation: {args.method}")
    if not single:
        print("  per-class AP:")
        for cls, res in sorted(per_class.items()):
            print(f"    class {cls}: AP={res['ap']:.4f}  "
                  f"TP={res['tp']} FP={res['fp']} "
                  f"GT={res['n_ground_truth']} "
                  f"max_recall={res['max_recall']:.3f}")
    else:
        res = next(iter(per_class.values()))
        print(f"  TP={res['tp']}  FP={res['fp']}  "
              f"GT={res['n_ground_truth']}  "
              f"max_recall={res['max_recall']:.3f}")
        if res["max_recall"] >= 0.999:
            print("  NOTE: recall reaches 1.0. Either the detector missed "
                  "nothing,\n        or the ground-truth count is wrong. "
                  "Verify which.")

    print(f"\n{name} = {map_value:.4f}")

    if single:
        print("\nOnly one class was found, so this is AP, not mAP. Report it "
              "as AP\nin Chapter 3 unless you add per-class (e.g. per-FDI-"
              "quadrant) labels.")

    if args.json_out:
        payload = {
            "map": map_value,
            "is_single_class": single,
            "method": args.method,
            "per_class": {
                str(k): {kk: vv for kk, vv in v.items()
                         if not isinstance(vv, np.ndarray)}
                for k, v in per_class.items()
            },
        }
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nwrote {args.json_out}")

    return map_value


if __name__ == "__main__":
    main()