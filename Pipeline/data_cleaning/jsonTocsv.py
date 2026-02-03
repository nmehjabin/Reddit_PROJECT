import json, csv

INFILE = "r_cybersecurity_posts.jsonl"
OUTFILE = "r_cybersecurity_posts_allcols.csv"

# 1) collect all keys
keys = set()
with open(INFILE, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        keys.update(obj.keys())

keys = sorted(keys)

# 2) write csv
with open(INFILE, "r") as fin, open(OUTFILE, "w", newline="", encoding="utf-8") as fout:
    writer = csv.DictWriter(fout, fieldnames=keys, extrasaction="ignore")
    writer.writeheader()
    for line in fin:
        line = line.strip()
        if not line:
            continue
        writer.writerow(json.loads(line))

print("done:", OUTFILE)



#json file validation check
# import json

# infile = "r_cybersecurity_posts.jsonl"

# with open(infile, "r") as f:
#     for i, line in enumerate(f, start=1):
#         line = line.strip()
#         if not line:
#             continue
#         try:
#             json.loads(line)
#         except json.JSONDecodeError as e:
#             print("First bad line:", i)
#             print("Error:", e)
#             print("Content:", line[:300])
#             break

# print("JSONL file validation completed.")