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
