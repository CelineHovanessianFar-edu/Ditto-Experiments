"""Task-8: Evaluate a saved Ditto checkpoint on the test set.

Usage (from the repo root):
    python Task-8/test.py --checkpoint checkpoints/Task-8/catalog_pairs/model.pt

    # With a custom threshold instead of tuning on valid set:
    python Task-8/test.py --checkpoint checkpoints/Task-8/catalog_pairs/model.pt --threshold 0.5
"""

import os
import sys
import argparse
import torch
import numpy as np
import sklearn.metrics as metrics

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "Snippext_public"))

from torch.utils import data
from ditto_light.dataset import DittoDataset
from ditto_light.ditto import DittoModel, evaluate

TASK_DIR = os.path.join(REPO_ROOT, "Task-8")
VALID_TXT = os.path.join(TASK_DIR, "valid.txt")
TEST_TXT = os.path.join(TASK_DIR, "test.txt")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved Ditto checkpoint on Task-8")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the saved model.pt checkpoint")
    parser.add_argument("--lm", type=str, default="distilbert",
                        help="Language model used during training")
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=None,
                        help="Fixed threshold for classification. "
                             "If not set, tunes on the validation set.")
    hp = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load model
    model = DittoModel(device=device, lm=hp.lm)
    ckpt = torch.load(hp.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()
    print(f"Loaded checkpoint from {hp.checkpoint} (epoch {ckpt.get('epoch', '?')})")

    padder = DittoDataset.pad

    # determine threshold
    if hp.threshold is not None:
        th = hp.threshold
        print(f"Using fixed threshold: {th}")
    else:
        print("Tuning threshold on validation set...")
        valid_dataset = DittoDataset(VALID_TXT, lm=hp.lm, max_len=hp.max_len)
        valid_iter = data.DataLoader(dataset=valid_dataset,
                                     batch_size=hp.batch_size,
                                     shuffle=False,
                                     num_workers=0,
                                     collate_fn=padder)
        valid_f1, th = evaluate(model, valid_iter)
        print(f"Validation F1: {valid_f1:.4f}  (best threshold: {th:.2f})")

    # evaluate on test set
    print("Evaluating on test set...")
    test_dataset = DittoDataset(TEST_TXT, lm=hp.lm, max_len=hp.max_len)
    test_iter = data.DataLoader(dataset=test_dataset,
                                batch_size=hp.batch_size,
                                shuffle=False,
                                num_workers=0,
                                collate_fn=padder)
    test_f1 = evaluate(model, test_iter, threshold=th)

    # detailed metrics
    all_probs = []
    all_y = []
    with torch.no_grad():
        for batch in test_iter:
            x, y = batch
            logits = model(x)
            probs = logits.softmax(dim=1)[:, 1]
            all_probs += probs.cpu().numpy().tolist()
            all_y += y.cpu().numpy().tolist()

    preds = [1 if p > th else 0 for p in all_probs]
    print("\n===== Test Results =====")
    print(f"Threshold : {th:.2f}")
    print(f"F1        : {test_f1:.4f}")
    print(f"Precision : {metrics.precision_score(all_y, preds):.4f}")
    print(f"Recall    : {metrics.recall_score(all_y, preds):.4f}")
    print(f"Accuracy  : {metrics.accuracy_score(all_y, preds):.4f}")
    print(f"Samples   : {len(all_y)}")


if __name__ == "__main__":
    main()
