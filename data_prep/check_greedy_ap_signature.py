import sys
from pathlib import Path
sys.path.insert(0, str(Path(r"D:\DentalCalib-Net\data_prep")))

import inspect
from compute_map import greedy_ap

print(inspect.signature(greedy_ap))
print(inspect.getsource(greedy_ap))