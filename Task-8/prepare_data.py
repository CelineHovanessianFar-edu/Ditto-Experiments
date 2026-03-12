"""Convert the catalog_pairs_token_leakless.parquet into Ditto-format txt files.

Ditto expects each line to be:
    <entity_left> \t <entity_right> \t <label>

where each entity is a sequence of  COL <col_name> VAL <col_value>  tokens.

Outputs: train.txt, valid.txt, test.txt  in the same directory.
"""

import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARQUET_PATH = os.path.join(SCRIPT_DIR, "catalog_pairs_token_leakless.parquet")
OUTPUT_DIR = SCRIPT_DIR


def row_to_entity(row, suffix):
    """Build a Ditto-style entity string from one side of a pair."""
    parts = []
    for col in ["displayname", "category"]:
        val = str(row[f"{col}_{suffix}"]).strip()
        parts.append(f"COL {col} VAL {val}")
    return " ".join(parts)


def convert():
    df = pd.read_parquet(PARQUET_PATH)
    print(f"Loaded {len(df)} rows")

    for split_name in ["train", "valid", "test"]:
        subset = df[df["split"] == split_name]
        out_path = os.path.join(OUTPUT_DIR, f"{split_name}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            for _, row in subset.iterrows():
                left = row_to_entity(row, "1")
                right = row_to_entity(row, "2")
                label = int(row["target"])
                f.write(f"{left}\t{right}\t{label}\n")
        print(f"  {split_name}: {len(subset)} pairs -> {out_path}")


if __name__ == "__main__":
    convert()
