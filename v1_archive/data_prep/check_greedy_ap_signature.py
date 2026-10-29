try:
    from .paths import project_path, raw_data_path
except ImportError:
    from paths import project_path, raw_data_path

import sys
from pathlib import Path
sys.path.insert(0, str(Path(project_path("data_prep"))))

import inspect
from compute_map import greedy_ap

print(inspect.signature(greedy_ap))
print(inspect.getsource(greedy_ap))