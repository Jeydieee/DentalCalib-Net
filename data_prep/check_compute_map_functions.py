import sys
from pathlib import Path
sys.path.insert(0, str(Path(r"D:\DentalCalib-Net\data_prep")))

import inspect
from compute_map import greedy_match, compute_ap

print("=== greedy_match ===")
print(inspect.signature(greedy_match))
print(inspect.getsource(greedy_match))

print("\n=== compute_ap ===")
print(inspect.signature(compute_ap))
print(inspect.getsource(compute_ap))