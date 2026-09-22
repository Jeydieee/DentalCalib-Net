import json
from pathlib import Path


def load_calibration_results(path):
    with open(path) as f:
        return json.load(f)


def compile_table_9(calibration_data, confidence_type='raw'):
    """Table 9: Baseline Calibration and Detection Accuracy of YOLOv8 and
    RT-DETR on the Clean DENTEX 2023 Test Set.
    Columns: Model | Architecture | ECE | MCE | D-ECE | RD Deviation (RMSD) | mAP@0.50
    Per the proposal, this is the CLEAN, baseline condition."""

    ARCHITECTURE_NAMES = {
        'yolov8': 'One-Stage Anchor-Free',
        'rtdetr': 'Transformer-Based',
    }

    rows = []
    for model in ['yolov8', 'rtdetr']:
        row = next((r for r in calibration_data
                    if r['model'] == model
                    and r['condition'] == 'clean'
                    and r['confidence_type'] == confidence_type), None)

        if row is None:
            print(f"WARNING: no matching row found for model={model}, condition=clean, confidence_type={confidence_type}")
            continue

        rows.append({
            'Model': model.upper() if model == 'yolov8' else 'RT-DETR',
            'Architecture': ARCHITECTURE_NAMES[model],
            'ECE': round(row['ece'], 4),
            'MCE': round(row['mce'], 4),
            'D-ECE': round(row['dece'], 4),
            'RD Deviation (RMSD)': round(row['rmsd'], 4),
            'mAP@0.50': round(row['map50'], 4),
        })

    return rows


def print_table(rows, title):
    print(f"\n{title}")
    print("-" * 100)
    if not rows:
        print("(no rows)")
        return

    headers = list(rows[0].keys())
    col_widths = {h: max(len(h), max(len(str(r[h])) for r in rows)) + 2 for h in headers}

    header_line = "".join(h.ljust(col_widths[h]) for h in headers)
    print(header_line)
    print("-" * len(header_line))

    for row in rows:
        print("".join(str(row[h]).ljust(col_widths[h]) for h in headers))


def save_table_csv(rows, path):
    import csv
    if not rows:
        return
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved -> {path}")


if __name__ == "__main__":
    CALIBRATION_RESULTS_PATH = r"D:\DentalCalib-Net\stage5_outputs\calibration_results.json"
    OUTPUT_DIR = r"D:\DentalCalib-Net\stage5_outputs\compiled_tables"

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    calibration_data = load_calibration_results(CALIBRATION_RESULTS_PATH)

    table_9 = compile_table_9(calibration_data, confidence_type='raw')

    print_table(table_9, "Table 9: Baseline Calibration and Detection Accuracy (Clean, Raw Confidence)")

    save_table_csv(table_9, f"{OUTPUT_DIR}/table_9.csv")