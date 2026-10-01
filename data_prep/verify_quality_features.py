try:
    from .paths import project_path, raw_data_path
except ImportError:
    from paths import project_path, raw_data_path

import json

with open(project_path("data_prep", "quality_features.json")) as f:
    data = json.load(f)

print("Partitions:", list(data.keys()))
print("Val image count:", len(data["val"]))
print("Test clean count:", len(data["test_clean"]))
print("OOD conditions:", len(data["test_ood"]))
print("Example val entry:", list(data["val"].items())[0])
print("Example OOD condition image count:", len(data["test_ood"]["gaussian_noise_S3"]))