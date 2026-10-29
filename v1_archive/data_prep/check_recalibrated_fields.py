try:
    from .paths import project_path, raw_data_path
except ImportError:
    from paths import project_path, raw_data_path

import json

with open(project_path("stage4_outputs", "recalibrated_predictions", "yolov8_clean_recalibrated.json")) as f:
    data = json.load(f)

print("Available keys in a sample prediction:")
print(list(data[0]['predictions'][0].keys()))