try:
    from .paths import project_path, raw_data_path
except ImportError:
    from paths import project_path, raw_data_path

REPOOL_JSON = project_path("data_prep", "repool_output", "dentex_merged_1005_resized.json")

import json

with open(REPOOL_JSON) as f:
    data = json.load(f)

print("Top-level keys:", list(data.keys()) if isinstance(data, dict) else "list")
if isinstance(data, dict) and "images" in data:
    print(f"\n{len(data['images'])} images. First 3 entries:")
    for img in data["images"][:3]:
        print(img)