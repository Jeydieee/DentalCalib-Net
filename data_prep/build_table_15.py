import csv
import numpy as np

CSV_PATH = r"D:\DentalCalib-Net\stage5_outputs\calibration_results.csv"

with open(CSV_PATH) as f:
    rows = list(csv.DictReader(f))

d = {(r["model"], r["condition"], r["confidence_type"]): r for r in rows}

CORRUPTIONS = ["gaussian_noise", "motion_blur", "brightness_variation", "jpeg_compression"]
SEVERITIES = [1, 2, 3, 4, 5]
MODEL = "rtdetr"


def get(condition, ct, metric):
    return float(d[(MODEL, condition, ct)][metric])


def print_row(label, condition):
    vals = []
    for ct in ["raw", "ts", "dcn"]:
        vals += [get(condition, ct, "ece"), get(condition, ct, "mce"),
                  get(condition, ct, "dece"), get(condition, ct, "map50")]
    row = f"{label:28s} |"
    for i in range(0, 12, 4):
        row += f" {vals[i]:8.4f}{vals[i+1]:8.4f}{vals[i+2]:8.4f}{vals[i+3]:8.4f} |"
    print(row)


print(f'{"Condition":28s} | {"--- Uncalibrated (raw) ---":40s} | {"--- Temperature Scaling ---":40s} | {"--- DentalCalib-Net ---":40s}')
print(f'{"":28s} | {"ECE":>8s}{"MCE":>8s}{"D-ECE":>8s}{"mAP50":>8s} | {"ECE":>8s}{"MCE":>8s}{"D-ECE":>8s}{"mAP50":>8s} | {"ECE":>8s}{"MCE":>8s}{"D-ECE":>8s}{"mAP50":>8s}')

print_row("Clean (Baseline)", "clean")
print()

agg = {ct: {m: [] for m in ["ece", "mce", "dece", "map50"]} for ct in ["raw", "ts", "dcn"]}

for corr in CORRUPTIONS:
    label_name = corr.replace("_", " ").title()
    for sev in SEVERITIES:
        cond = f"{corr}_S{sev}"
        print_row(f"{label_name} S{sev}", cond)
        for ct in ["raw", "ts", "dcn"]:
            for m in ["ece", "mce", "dece", "map50"]:
                agg[ct][m].append(get(cond, ct, m))
    print()

row = f'{"Average (All Severities, 4 OOD)":28s} |'
for ct in ["raw", "ts", "dcn"]:
    row += f' {np.mean(agg[ct]["ece"]):8.4f}{np.mean(agg[ct]["mce"]):8.4f}{np.mean(agg[ct]["dece"]):8.4f}{np.mean(agg[ct]["map50"]):8.4f} |'
print(row)