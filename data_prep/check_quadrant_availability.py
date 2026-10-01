try:
    from .paths import project_path, raw_data_path
except ImportError:
    from paths import project_path, raw_data_path

import json

with open(project_path("stage4_outputs", "labeled_predictions", "yolov8_predictions_clean_labeled.json")) as f:
    data = json.load(f)

# find one correctly-labeled prediction (label==1) and show its full structure
for entry in data:
    for pred in entry['predictions']:
        if pred['label'] == 1:
            print(pred)
            break
    else:
        continue
    break