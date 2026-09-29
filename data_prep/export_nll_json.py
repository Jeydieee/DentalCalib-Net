CSV_PATH = r"D:\DentalCalib-Net\stage5_outputs\nll_results.csv"
OUT_PATH = r"D:\DentalCalib-Net\frontend\data\nll_results.json"

import csv
import json

with open(CSV_PATH) as f:
    rows = list(csv.DictReader(f))

output = {}
for r in rows:
    key = f"{r['model']}__{r['condition']}"
    output[key] = {
        "nll_raw": float(r["nll_raw"]),
        "nll_ts": float(r["nll_ts"]),
    }

with open(OUT_PATH, "w") as f:
    json.dump(output, f)

print(f"wrote {len(output)} entries -> {OUT_PATH}")