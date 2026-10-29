import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_ROOT = Path(os.environ.get("DENTEX_RAW_DATA_ROOT", PROJECT_ROOT / "DENTEX 2023"))


def project_path(*parts):
    return str(PROJECT_ROOT.joinpath(*parts))


def raw_data_path(*parts):
    return str(RAW_DATA_ROOT.joinpath(*parts))