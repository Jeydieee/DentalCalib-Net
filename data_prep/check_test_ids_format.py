try:
    from .paths import project_path, raw_data_path
except ImportError:
    from paths import project_path, raw_data_path

TEST_IDS_CSV = project_path("data_prep", "split_output", "test_ids.csv")

with open(TEST_IDS_CSV) as f:
    lines = f.readlines()

print(f"Total lines: {len(lines)}")
print("First 10 lines, raw:")
for line in lines[:10]:
    print(repr(line))