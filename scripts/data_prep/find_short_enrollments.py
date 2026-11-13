import json

with open("output/dataset_magicodec_131k_60s_metadata.jsonl", "r") as f:
    metadata = [json.loads(line) for line in f]

short_enrollments = [m for m in metadata if m["tv_end_secs"] - m["tv_start_secs"] < 3.0]

print(f"Found {len(short_enrollments)} short enrollments.")
for m in short_enrollments:
    print(m)