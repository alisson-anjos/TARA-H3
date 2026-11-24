"""Aggregate metrics.csv into per-category tables (mean ± std over clips and seeds).

    python research/eval/summarize.py metrics.csv > results.md
"""
import sys

import pandas as pd

df = pd.read_csv(sys.argv[1])
df["pic_last"] = df["pic_3"]
df["ref_last"] = df["ref_3"]
df["collapse"] = (df["ref_3"] > df["pic_3"]).astype(float)  # the last third looks more like the reference person
cols = ["timing_mean", "yaw_r", "mouth_r", "flow_r", "pic_1", "pic_last", "ref_last", "collapse"]
print("| category | variant | n | " + " | ".join(cols) + " |")
print("|---|---|---:|" + "---:|" * len(cols))
for (cat, var), g in df.groupby(["category", "variant"]):
    cells = [f"{g[c].mean():.2f} ± {g[c].std():.2f}" if g[c].notna().sum() > 1 else f"{g[c].mean():.2f}" for c in cols]
    print(f"| {cat} | {var} | {len(g)} | " + " | ".join(cells) + " |")
