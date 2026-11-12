"""Step 2 (already done - prompts/splits.json is in the repo): split the 345 Quick, Draw! objects into
199 train / 66 val / 66 test, after dropping 14 that make bad drawing prompts (see prompts/exclusions.tsv).

    python scripts/02_make_splits.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from svgbudget.prompts import make_splits

categories_file = sorted(Path("prompts").glob("quickdraw_categories@*.txt"))[-1]
categories = [l.strip() for l in open(categories_file) if l.strip()]
exclude = [l.split("\t")[0] for l in open("prompts/exclusions.tsv") if l.strip()]
splits = make_splits(categories, exclude, seed=0)
splits["meta"] = {"categories_file": str(categories_file), "n_excluded": len(exclude),
                  "fractions": [0.6, 0.2, 0.2], "seed": 0}
json.dump(splits, open("prompts/splits.json", "w"), indent=2)
print({k: len(splits[k]) for k in ("train", "val", "test")})
