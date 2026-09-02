import json
import re
from pathlib import Path
from collections import Counter

def parse_label(label):
    """'1-çürük-27' -> diagnosis term + FDI tooth number"""
    parts = label.split('-')
    if len(parts) != 3:
        return None, None
    _, diag_term, fdi = parts
    return diag_term, fdi

def scan_vocabulary(test_dir):
    test_dir = Path(test_dir)
    files = sorted(test_dir.glob('test_*.json'), key=lambda p: int(re.search(r'\d+', p.stem).group()))

    term_counts = Counter()
    term_files = {}          # term -> set of files it appears in (for spot-checking later)
    unparseable = []

    for fpath in files:
        with open(fpath, encoding='utf-8') as f:
            data = json.load(f)

        for shape in data['shapes']:
            diag_term, fdi = parse_label(shape['label'])
            if diag_term is None:
                unparseable.append((fpath.name, shape['label']))
                continue
            term_counts[diag_term] += 1
            term_files.setdefault(diag_term, set()).add(fpath.name)

    print(f"Scanned {len(files)} files.\n")
    print("Full term vocabulary (most common first):")
    for term, count in term_counts.most_common():
        n_files = len(term_files[term])
        print(f"  {term}: {count} occurrences across {n_files} files")

    if unparseable:
        print(f"\n{len(unparseable)} unparseable labels:")
        for fname, label in unparseable[:20]:  # show first 20 only
            print(f"  {fname}: '{label}'")
        if len(unparseable) > 20:
            print(f"  ... and {len(unparseable) - 20} more")

    return term_counts

if __name__ == "__main__":
    scan_vocabulary(test_dir="D:\\DENTEX 2023\\test_data\\disease\\label")