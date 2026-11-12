"""Step 6: evaluate a model - the same pipeline for the baseline, every SFT model and the RL model.

    python scripts/06_evaluate.py --name eval_baseline                               # unseen objects (val)
    python scripts/06_evaluate.py --name eval_baseline_seen --split train --n-samples 1
    python scripts/06_evaluate.py --name eval_rl_v1 --adapter runs/<stamp>_rl_v1/adapter
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from svgbudget.evaluate import gate_all, generate_all, judge_all, summarize
from svgbudget.judges import judge_paths
from svgbudget.model import load_model
from svgbudget.prompts import PROMPT_VERSION, all_candidates
from svgbudget.runlog import start_run

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--split", default="val", choices=["train", "val", "test"])
ap.add_argument("--budgets", type=int, nargs="+", default=[3, 6, 12])
ap.add_argument("--n-samples", type=int, default=4)
ap.add_argument("--adapter", default=None, help="a LoRA adapter folder (merged into the base model)")
ap.add_argument("--thinking", action="store_true", help="let Qwen reason before answering")
ap.add_argument("--max-new-tokens", type=int, default=None, help="default 1024 (8192 with --thinking)")
ap.add_argument("--batch-prompts", type=int, default=24)
args = ap.parse_args()
if args.split == "test":
    sys.exit("The test split is reserved for a final evaluation.")

# Qwen's recommended sampling settings: non-thinking T=0.7 / top-p 0.8; thinking T=0.6 / top-p 0.95; top-k 20
sampling = dict(temperature=0.6, top_p=0.95) if args.thinking else dict(temperature=0.7, top_p=0.8)
cfg = dict(n_samples=args.n_samples, max_new_tokens=args.max_new_tokens or (8192 if args.thinking else 1024),
           top_k=20, enable_thinking=args.thinking, **sampling)
splits = json.load(open("prompts/splits.json"))
candidates = all_candidates(splits)
tasks = [(c, n) for c in splits[args.split] for n in args.budgets]
judges = judge_paths()

run = start_run(args.name, {**vars(args), "generation": cfg, "prompt_version": PROMPT_VERSION,
                            "n_tasks": len(tasks), "judges": judges})
t0 = time.time()
model, tok = load_model(adapter=args.adapter)
records = generate_all(model, tok, tasks, cfg, args.batch_prompts)
del model
t_gen = time.time() - t0
images = gate_all(records)
(run / "renders").mkdir()
for i, img in enumerate(images):
    if img is not None:
        img.save(run / "renders" / f"{i}.png")
judge_all(records, images, judges, candidates)
summary = summarize(records, list(judges), len(candidates))
with open(run / "generations.jsonl", "w") as f:
    for r in records:
        f.write(json.dumps(r) + "\n")
(run / "summary.json").write_text(json.dumps({**summary, "generation_seconds": round(t_gen)}, indent=2))

lines = [f"{'group':6s} {'n':>4s} {'gate':>6s} {'trunc':>5s}" + "".join(f" | {j:>8s} top1  top5  vs-human" for j in judges)]
for g, s in summary.items():
    row = f"{g:6s} {s['n']:4d} {100 * s['gate_pass']:5.1f}% {s['truncated']:5d}"
    for j in judges:
        row += f" | {100 * s[j]['usable_top1']:12.1f}% {100 * s[j]['usable_top5']:4.1f}% {100 * s[j]['usable_top1_vs_human']:7.1f}%"
    lines.append(row)
lines += [f"failure reasons: {summary['all']['reasons']}", f"generation time: {t_gen:.0f}s", f"run folder: {run}"]
(run / "summary.txt").write_text("\n".join(lines))
print("\n".join(lines))
