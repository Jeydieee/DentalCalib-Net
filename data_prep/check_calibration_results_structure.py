import json

with open(r"D:\DentalCalib-Net\stage5_outputs\calibration_results.json") as f:
    data = json.load(f)

print(f"Total rows: {len(data)}")
print(f"Example row: {data[0]}")
print(f"Unique conditions: {sorted(set(row.get('condition', row.get('severity', '?')) for row in data))[:10]}")