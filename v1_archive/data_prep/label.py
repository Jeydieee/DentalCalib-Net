import json
from collections import defaultdict, Counter

def load_merged(path='dentex_merged_1005.json'):
    with open(path) as f:
        return json.load(f)

def count_impacted(merged):
    counts = defaultdict(int)
    for ann in merged['annotations']:
        if ann['category_id_3'] == 0:  # 0 = Impacted
            counts[ann['image_id']] += 1
    return counts

def bucket(n):
    if n == 0: return '0'
    if n == 1: return '1'
    return '2+'

def build_labels(merged, counts):
    image_ids = [img['id'] for img in merged['images']]
    y = [bucket(counts[iid]) for iid in image_ids]
    return image_ids, y

def main(merged_path='dentex_merged_1005.json', out_path='strat_labels.json'):
    merged = load_merged(merged_path)
    counts = count_impacted(merged)
    image_ids, y = build_labels(merged, counts)

    dist = Counter(y)
    print("Bucket distribution:", dict(dist))

    if dist.get('2+', 0) < 30:
        print("WARNING: '2+' bucket has <30 images. A three-way 60/20/20 stratified "
              "split may error or produce a near-meaningless test stratum. "
              "Consider collapsing to binary (has impaction / doesn't) before Phase 3.")

    with open(out_path, 'w') as f:
        json.dump({'image_ids': image_ids, 'labels': y}, f)
    print(f"Saved labels -> {out_path}")

if __name__ == "__main__":
    main()