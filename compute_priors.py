"""
Computes each EMOTIC category's prevalence (fraction of training rows where
it's positive) and writes calibration.json for inference.py to use.

This reproduces the emotion_prevalence() call from the training notebook
(cell 22) -- that function itself wasn't saved in the notebook, but its
job is simple: for each of the 26 categories, what fraction of training
examples had that label set to 1? That's the same count the notebook's
calculate_emotion_pos_weights() (cell 16) already computes, just expressed
as a rate instead of a pos_weight ratio.

Run this from wherever you have the EMOTIC annotations, e.g.:

    python compute_priors.py --annot-dir emotic/annots_arrs --max-rows 5000

--max-rows should match whatever MAX_ROWS["train"] was set to when the
deployed model was actually trained (see the notebook's cell 12), so the
priors reflect the same data distribution the model learned from. Leave
it unset to use every row in annot_arrs_train.csv.

Output: calibration.json, one prior (0-1) per category. Copy it next to
inference.py in the app -- if it's missing, the app just skips calibration
and falls back to raw scores like before.
"""

import argparse
import csv
import json
from pathlib import Path

EMOTIONS = [
    "Peace", "Affection", "Esteem", "Anticipation", "Engagement",
    "Confidence", "Happiness", "Pleasure", "Excitement", "Surprise",
    "Sympathy", "Doubt/Confusion", "Disconnection", "Fatigue",
    "Embarrassment", "Yearning", "Disapproval", "Aversion", "Annoyance",
    "Anger", "Sensitivity", "Sadness", "Disquietment", "Fear", "Pain",
    "Suffering",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annot-dir", required=True, help="folder containing annot_arrs_train.csv")
    ap.add_argument("--max-rows", type=int, default=None, help="match MAX_ROWS['train'] used for training")
    ap.add_argument("--out", default="calibration.json")
    args = ap.parse_args()

    csv_path = Path(args.annot_dir) / "annot_arrs_train.csv"
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if args.max_rows is not None:
        rows = rows[: args.max_rows]

    total = len(rows)
    if total == 0:
        raise SystemExit(f"No rows read from {csv_path}")

    priors = {}
    for name in EMOTIONS:
        positive = sum(float(row[name]) for row in rows)
        # Clamp away from exactly 0 or 1 so log-odds in inference.py never
        # divides by zero for a category that never (or always) appears
        # in this many rows.
        rate = positive / total
        priors[name] = min(max(rate, 1e-4), 1 - 1e-4)

    Path(args.out).write_text(json.dumps(priors, indent=2))
    print(f"Wrote {args.out} from {total} training rows")
    for name, p in sorted(priors.items(), key=lambda kv: -kv[1]):
        print(f"  {name:20s} {p:.4f}")


if __name__ == "__main__":
    main()
