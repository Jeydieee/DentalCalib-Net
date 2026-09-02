import json
import re
from pathlib import Path
from collections import Counter

# Turkish diagnosis term -> DENTEX category_id_3
DIAG_MAP = {
    'çürük': 1,   # Caries
    'gömülü': 0,  # Impacted
    # 'kanal' and 'küretaj' are NOT diagnosis categories - dropped
}

FDI_QUADRANT_MAP = {'1': 0, '2': 1, '3': 2, '4': 3}  # FDI quadrant digit -> category_id_1

def parse_label(label):
    """'1-çürük-27' -> diagnosis term + FDI tooth number"""
    parts = label.split('-')
    if len(parts) != 3:
        return None, None
    _, diag_term, fdi = parts
    return diag_term, fdi

def fdi_to_ids(fdi_str):
    """'27' -> (quadrant_id, tooth_id) matching category_id_1/category_id_2"""
    if fdi_str is None or len(fdi_str) != 2:
        return None, None
    quadrant_digit, tooth_digit = fdi_str[0], fdi_str[1]
    quadrant_id = FDI_QUADRANT_MAP.get(quadrant_digit)
    if not tooth_digit.isdigit():
        return quadrant_id, None
    tooth_id = int(tooth_digit) - 1  # tooth 1-8 -> 0-7
    return quadrant_id, tooth_id

def polygon_to_bbox(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, y_min = min(xs), min(ys)
    w, h = max(xs) - x_min, max(ys) - y_min
    return [x_min, y_min, w, h]

def convert_test_folder(test_dir, out_path='test_quadrant_enumeration_disease.json'):
    test_dir = Path(test_dir)
    files = sorted(test_dir.glob('test_*.json'), key=lambda p: int(re.search(r'\d+', p.stem).group()))

    images = []
    annotations = []
    ann_id = 1
    skipped = 0
    dropped_terms = Counter()          # counts WHY things were dropped
    parse_failures = 0                 # labels that didn't split into 3 parts at all

    for img_id, fpath in enumerate(files, start=1):
        with open(fpath, encoding='utf-8') as f:
            data = json.load(f)

        images.append({
            'id': img_id,
            'file_name': data['imagePath'],
            'height': data['imageHeight'],
            'width': data['imageWidth'],
        })

        for shape in data['shapes']:
            diag_term, fdi = parse_label(shape['label'])

            if diag_term is None:
                # label didn't match the expected "N-term-NN" format at all
                parse_failures += 1
                dropped_terms[f"UNPARSEABLE: {shape['label']}"] += 1
                skipped += 1
                continue

            if diag_term not in DIAG_MAP:
                dropped_terms[diag_term] += 1
                skipped += 1
                continue

            quadrant_id, tooth_id = fdi_to_ids(fdi)
            if quadrant_id is None or tooth_id is None:
                dropped_terms[f"BAD_FDI: {fdi} (term={diag_term})"] += 1
                skipped += 1
                continue

            bbox = polygon_to_bbox(shape['points'])
            annotations.append({
                'id': ann_id,
                'image_id': img_id,
                'bbox': bbox,
                'category_id_1': quadrant_id,
                'category_id_2': tooth_id,
                'category_id_3': DIAG_MAP[diag_term],
                'iscrowd': 0,
                'area': bbox[2] * bbox[3],
                'segmentation': [[coord for point in shape['points'] for coord in point]],
            })
            ann_id += 1

    merged = {
        'images': images,
        'annotations': annotations,
        'categories_1': [{'id': i, 'name': str(i+1), 'supercategory': str(i+1)} for i in range(4)],
        'categories_2': [{'id': i, 'name': str(i+1), 'supercategory': str(i+1)} for i in range(8)],
        'categories_3': [
            {'id': 0, 'name': 'Impacted', 'supercategory': 'Impacted'},
            {'id': 1, 'name': 'Caries', 'supercategory': 'Caries'},
            {'id': 2, 'name': 'Periapical Lesion', 'supercategory': 'Periapical Lesion'},
            {'id': 3, 'name': 'Deep Caries', 'supercategory': 'Deep Caries'},
        ],
    }

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False)

    total_shapes = len(annotations) + skipped
    print(f"Converted {len(images)} images, {len(annotations)} annotations, {skipped} shapes dropped")
    print(f"  of which {parse_failures} failed to parse as 'N-term-NN' at all")
    print(f"  drop rate: {skipped}/{total_shapes} = {skipped/total_shapes:.1%}")
    print("\nDropped term breakdown (most common first):")
    for term, count in dropped_terms.most_common():
        print(f"  {term}: {count}")
    print(f"\nSaved -> {out_path}")

if __name__ == "__main__":
    convert_test_folder(test_dir="D:\\DENTEX 2023\\test_data\\disease\\label")