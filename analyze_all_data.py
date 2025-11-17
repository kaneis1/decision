import argparse
from pathlib import Path

import pandas as pd


Descriptor_cols = [
    "period","risk", "error", "delta", "infin", "contin", "r1", "r2", "r", "s", "t", "p"
]


def main():
    data_path = "data/all_data.csv"
    df = pd.read_csv(data_path)
    groups = []
    start_idx = 0
    for i in range(0, len(df["data_id"]) - 1):
        # If the next period is not equal to current period + 1, cut at i
        if df["period"].iloc[i] != df["period"].iloc[i + 1] - 1:
            # Only selectpisodes that start from period 1 (can adjust as needed)       
            groups.append(df.iloc[start_idx:i+1].copy())
            start_idx = i+1
    groups.append(df.iloc[start_idx:].copy())

    for ep in groups:
        print(len(ep))

            
            
if __name__ == "__main__":
    main()

