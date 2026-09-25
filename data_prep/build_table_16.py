CSV_PATH = r"D:\DentalCalib-Net\stage5_outputs\calibration_results.csv"

import csv
import numpy as np

with open(CSV_PATH) as f:
    rows = list(csv.DictReader(f))

d = {(r["model"], r["condition"], r["confidence_type"]): r for r in rows}

CORRUPTIONS = ["gaussian_noise", "motion_blur", "brightness_variation", "jpeg_compression"]
SEVERITIES = [1, 2, 3, 4, 5]
OOD_CONDITIONS = [f"{c}_S{s}" for c in CORRUPTIONS for s in SEVERITIES]
MODELS = ["yolov8", "rtdetr"]


def get(model, condition, ct, metric):
    return float(d[(model, condition, ct)][metric])


print("Table 16: ECE/MCE/D-ECE Reduction Magnitude, mAP@0.50 Safety Invariant")
print("mAP Safety Invariant = # of 20 OOD conditions where recalibrated mAP50 >= raw mAP50")
print("(per Appendix 1: 'confirming that neither recalibration method will reduce")
print(" mAP@0.50 below the pre-recalibration baseline')")
print()

for model in MODELS:
    raw_ece = np.mean([get(model, c, "raw", "ece") for c in OOD_CONDITIONS])
    ts_ece = np.mean([get(model, c, "ts", "ece") for c in OOD_CONDITIONS])
    dcn_ece = np.mean([get(model, c, "dcn", "ece") for c in OOD_CONDITIONS])
    raw_mce = np.mean([get(model, c, "raw", "mce") for c in OOD_CONDITIONS])
    ts_mce = np.mean([get(model, c, "ts", "mce") for c in OOD_CONDITIONS])
    dcn_mce = np.mean([get(model, c, "dcn", "mce") for c in OOD_CONDITIONS])
    raw_dece = np.mean([get(model, c, "raw", "dece") for c in OOD_CONDITIONS])
    ts_dece = np.mean([get(model, c, "ts", "dece") for c in OOD_CONDITIONS])
    dcn_dece = np.mean([get(model, c, "dcn", "dece") for c in OOD_CONDITIONS])

    ts_held = sum(1 for c in OOD_CONDITIONS if get(model, c, "ts", "map50") >= get(model, c, "raw", "map50"))
    dcn_held = sum(1 for c in OOD_CONDITIONS if get(model, c, "dcn", "map50") >= get(model, c, "raw", "map50"))

    print(f"[{model}]")
    print(f"  ECE Reduction        -- TS: {raw_ece-ts_ece:+.4f}   DCN: {raw_ece-dcn_ece:+.4f}")
    print(f"  MCE Reduction        -- TS: {raw_mce-ts_mce:+.4f}   DCN: {raw_mce-dcn_mce:+.4f}")
    print(f"  D-ECE Reduction      -- TS: {raw_dece-ts_dece:+.4f}   DCN: {raw_dece-dcn_dece:+.4f}")
    print(f"  mAP Safety Invariant -- TS: {ts_held}/20 held         DCN: {dcn_held}/20 held")
    print()