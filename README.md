# DentalCalib-Net

_*Thesis Writing 2 - Group 8*_

## Data paths

Scripts resolve files inside this repository relative to the repository root, so the checkout can live at any path on any operating system. The original DENTEX image folders are external and are not included in the repository. Set `DENTEX_RAW_DATA_ROOT` to the directory containing `training_data`, `validation_data`, and `test_data` before running scripts that read original images. If unset, scripts look for a `DENTEX 2023` directory in the repository root.

PowerShell example:

```powershell
$env:DENTEX_RAW_DATA_ROOT = "D:\datasets\DENTEX 2023"
```

For Bash:

```bash
export DENTEX_RAW_DATA_ROOT="/datasets/DENTEX 2023"
```

## Frontend auditor setup

The auditor displays precomputed predictions; it does not run YOLOv8 or RT-DETR on arbitrary uploads. Only the 201 supported DENTEX test images, using their original filenames, have saved predictions.

The generated auditor assets are excluded from Git by `.gitignore`. A fresh clone needs copies of the complete `frontend/data/` and `frontend/images/` folders from a shared drive or from a teammate who has generated them. Keep their existing directory structure. Without those assets, the auditor cannot validate uploads or show predictions.

Serve the frontend over HTTP so browsers can fetch its JSON data files:

```powershell
python -m http.server 8000 --directory frontend
```

Then open `http://localhost:8000`. Opening `index.html` directly as a `file://` URL may prevent the browser from loading the JSON files.
