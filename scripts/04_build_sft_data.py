"""Step 4: turn human sketches into SFT examples (prompt -> SVG).

Every version in the write-up is one command:
    python scripts/04_build_sft_data.py --name v1    --per-category 100                      # SFT v1
    python scripts/04_build_sft_data.py --name v2a   --per-category 100  --eps 2             # simplified strokes
    python scripts/04_build_sft_data.py --name v2b   --per-category 100  --max-points 40     # no long strokes
    python scripts/04_build_sft_data.py --name v3_300  --per-category 300  --eps 2           # 3x data
    python scripts/04_build_sft_data.py --name v3_1000 --per-category 1000 --eps 2           # 10x data
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from svgbudget.gate import check
from svgbudget.runlog import start_run
from svgbudget.sketches import build_sft_examples, sketch_to_compact_svg
from svgbudget.stats import length_stats

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--per-category", type=int, default=100, help="use the first K sketches of each training object")
ap.add_argument("--eps", type=float, default=0, help="stroke simplification tolerance (0 = off)")
ap.add_argument("--max-points", type=int, default=None, help="drop sketches with a longer stroke")
ap.add_argument("--token-cap", type=int, default=900)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(json.load(open("models/model.json"))["local_path"])
run = start_run(f"sft_build_{args.name}", vars(args))
out = Path("data/sft") / args.name
out.mkdir(parents=True, exist_ok=True)


def to_svg(drawing):
    return sketch_to_compact_svg(drawing, eps=args.eps)


def n_tokens(svg):
    return len(tok(svg, add_special_tokens=False).input_ids)


def passes_gate(svg, n):
    return check(svg, n)["ok"]


summary = {}
for name, path, k in [("examples", "data/quickdraw/train.jsonl", args.per_category),
                      ("dev_valcats", "data/quickdraw/val.jsonl", None)]:
    per_cat = defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        if k is None or len(per_cat[r["category"]]) < k:
            per_cat[r["category"]].append(r)
    rows = [r for c in sorted(per_cat) for r in per_cat[c]]
    examples, stats = build_sft_examples(rows, to_svg, n_tokens, passes_gate, token_cap=args.token_cap,
                                         seed=args.seed, max_points=args.max_points)
    with open(out / f"{name}.jsonl", "w") as f:
        for e in examples:
            f.write(json.dumps(e) + "\n")
    summary[name] = {**stats, "total": len(examples), "tokens": length_stats([e["tokens"] for e in examples])}
(run / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=1))
