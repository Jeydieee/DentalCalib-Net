try:
    from .paths import project_path, raw_data_path
except ImportError:
    from paths import project_path, raw_data_path

import sys
from pathlib import Path
sys.path.insert(0, str(Path(project_path("data_prep"))))

import inspect
from compute_map import greedy_match, compute_ap

print("=== greedy_match ===")
print(inspect.signature(greedy_match))
print(inspect.getsource(greedy_match))

print("\n=== compute_ap ===")
print(inspect.signature(compute_ap))
print(inspect.getsource(compute_ap))