"""Task-8: Train and evaluate Ditto on the catalog_pairs dataset.

Usage (from the repo root):
    python Task-8/run.py [OPTIONS]

This script:
  1. Converts the parquet to Ditto-format txt files (if not already done).
  2. Trains the Ditto model on train.txt, validates on valid.txt,
     and reports test F1 on test.txt.

All standard Ditto hyper-parameters can be overridden via CLI flags.
"""

import os
import sys
import argparse
import json
import random
import numpy as np
import torch

# ---- Ensure we can import ditto_light regardless of cwd ----
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "Snippext_public"))

from ditto_light.dataset import DittoDataset
from ditto_light.summarize import Summarizer
from ditto_light.knowledge import *
from ditto_light.ditto import train

# ---------- paths ----------------------------------------------------------
TASK_DIR = os.path.join(REPO_ROOT, "Task-8")
PARQUET = os.path.join(TASK_DIR, "catalog_pairs_token_leakless.parquet")
TRAIN_TXT = os.path.join(TASK_DIR, "train.txt")
VALID_TXT = os.path.join(TASK_DIR, "valid.txt")
TEST_TXT  = os.path.join(TASK_DIR, "test.txt")

TASK_NAME = "Task-8/catalog_pairs"


# ---------- data preparation -----------------------------------------------
def prepare_data():
    """Convert parquet -> Ditto txt files (idempotent)."""
    if os.path.exists(TRAIN_TXT) and os.path.exists(VALID_TXT) and os.path.exists(TEST_TXT):
        print("Ditto-format txt files already exist, skipping conversion.")
        return

    import pandas as pd
    df = pd.read_parquet(PARQUET)
    print(f"Loaded {len(df)} rows from parquet")

    def row_to_entity(row, suffix):
        parts = []
        for col in ["displayname", "category"]:
            val = str(row[f"{col}_{suffix}"]).strip()
            parts.append(f"COL {col} VAL {val}")
        return " ".join(parts)

    for split_name, path in [("train", TRAIN_TXT), ("valid", VALID_TXT), ("test", TEST_TXT)]:
        subset = df[df["split"] == split_name]
        with open(path, "w", encoding="utf-8") as f:
            for _, row in subset.iterrows():
                left = row_to_entity(row, "1")
                right = row_to_entity(row, "2")
                label = int(row["target"])
                f.write(f"{left}\t{right}\t{label}\n")
        print(f"  {split_name}: {len(subset)} pairs -> {path}")


# ---------- main -----------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train Ditto on Task-8 catalog pairs")
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--finetuning", dest="finetuning", action="store_true")
    parser.add_argument("--save_model", dest="save_model", action="store_true")
    parser.add_argument("--logdir", type=str, default="checkpoints/")
    parser.add_argument("--lm", type=str, default="distilbert")
    parser.add_argument("--fp16", dest="fp16", action="store_true")
    parser.add_argument("--da", type=str, default=None)
    parser.add_argument("--alpha_aug", type=float, default=0.8)
    parser.add_argument("--dk", type=str, default=None)
    parser.add_argument("--summarize", dest="summarize", action="store_true")
    parser.add_argument("--size", type=int, default=None)
    hp = parser.parse_args()

    # ---- seeds ----
    seed = hp.run_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ---- data preparation ----
    prepare_data()

    # ---- paths ----
    trainset = TRAIN_TXT
    validset = VALID_TXT
    testset  = TEST_TXT

    # ---- optional summarization ----
    task_config = {
        "name": TASK_NAME,
        "task_type": "classification",
        "vocab": ["0", "1"],
        "trainset": trainset,
        "validset": validset,
        "testset": testset,
    }

    if hp.summarize:
        summarizer = Summarizer(task_config, lm=hp.lm)
        trainset = summarizer.transform_file(trainset, max_len=hp.max_len)
        validset = summarizer.transform_file(validset, max_len=hp.max_len)
        testset  = summarizer.transform_file(testset, max_len=hp.max_len)

    # ---- optional domain knowledge injection ----
    if hp.dk is not None:
        if hp.dk == "product":
            injector = ProductDKInjector(task_config, hp.dk)
        else:
            injector = GeneralDKInjector(task_config, hp.dk)
        trainset = injector.transform_file(trainset)
        validset = injector.transform_file(validset)
        testset  = injector.transform_file(testset)

    # ---- build run tag ----
    run_tag = "%s_lm=%s_da=%s_dk=%s_su=%s_size=%s_id=%d" % (
        TASK_NAME, hp.lm, hp.da, hp.dk, hp.summarize, str(hp.size), hp.run_id
    )
    run_tag = run_tag.replace("/", "_")

    # ---- load datasets ----
    train_dataset = DittoDataset(trainset, lm=hp.lm, max_len=hp.max_len,
                                 size=hp.size, da=hp.da)
    valid_dataset = DittoDataset(validset, lm=hp.lm)
    test_dataset  = DittoDataset(testset, lm=hp.lm)

    hp.task = TASK_NAME  # needed by train() for checkpoint saving

    # ---- train & evaluate ----
    train(train_dataset, valid_dataset, test_dataset, run_tag, hp)


if __name__ == "__main__":
    main()
