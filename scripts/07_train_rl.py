"""Step 7: RL from the best SFT adapter. Reward = SigLIP 2's probability for the true object (0 if the gate
fails); CLIP is the independent check. Trains on training objects only.

    python scripts/07_train_rl.py --name rl_v1 --start sft_v3_1000
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from svgbudget.judges import judge_paths, load_judge
from svgbudget.prompts import PROMPT_VERSION, all_candidates
from svgbudget.rl import load_policy, train_rl
from svgbudget.runlog import latest_run, start_run

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="rl_v1")
ap.add_argument("--start", default="sft_v3_1000", help="the SFT run whose adapter RL starts from")
ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--prompts-per-step", type=int, default=16)
ap.add_argument("--group-size", type=int, default=8)
ap.add_argument("--lr", type=float, default=2e-6)
ap.add_argument("--warmup", type=int, default=10)
ap.add_argument("--eval-every", type=int, default=25)
ap.add_argument("--save-every", type=int, default=50)
ap.add_argument("--eval-tasks", type=int, default=66, help="objects per quick-eval set (budget 6, 2 drawings each)")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

splits = json.load(open("prompts/splits.json"))
candidates = all_candidates(splits)
start_adapter = latest_run(args.start) / "adapter"
rng = random.Random(1234)
eval_sets = {"val": [(c, 6) for c in rng.sample(splits["val"], min(args.eval_tasks, len(splits["val"])))],
             "train": [(c, 6) for c in rng.sample(splits["train"], args.eval_tasks)]}
run = start_run(args.name, {**vars(args), "start_adapter": str(start_adapter), "prompt_version": PROMPT_VERSION,
                            "reward": "siglip2 probability of the true object over 331; 0 if the gate fails",
                            "check_judge": "clip", "advantage": "reward - group mean (no std)", "kl": "none",
                            "rollout_sampling": "T=1.0, top_p=1.0, top_k=0", "eval_sets": eval_sets})
model, tok = load_policy(str(start_adapter))
paths = judge_paths()
reward_judge = load_judge(paths["siglip2"], candidates)
check_judge = load_judge(paths["clip"], candidates)
train_rl(model, tok, reward_judge, check_judge, splits["train"], eval_sets, run, steps=args.steps,
         prompts_per_step=args.prompts_per_step, group_size=args.group_size, lr=args.lr, warmup=args.warmup,
         eval_every=args.eval_every, save_every=args.save_every, seed=args.seed)
print(f"done | adapter saved in {run / 'adapter'}")
