try:
    from .paths import project_path, raw_data_path
except ImportError:
    from paths import project_path, raw_data_path

import json

with open(project_path("stage5_outputs", "calibration_results.json")) as f:
    data = json.load(f)

print(f"Total rows: {len(data)}")
print(f"Example row: {data[0]}")
print(f"Unique conditions: {sorted(set(row.get('condition', row.get('severity', '?')) for row in data))[:10]}")