"""Step 5: LoRA SFT on one of the datasets from step 4.

    python scripts/05_train_sft.py --name sft_v3_1000 --data v3_1000 --max-dev 500
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from svgbudget.model import load_model
from svgbudget.prompts import PROMPT_VERSION
from svgbudget.runlog import start_run
from svgbudget.sft import add_lora, is_dev, tokenize_example, train_sft

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--data", required=True, help="a folder name under data/sft/")
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--batch-size", type=int, default=16)
ap.add_argument("--micro-batch", type=int, default=4)
ap.add_argument("--lora-r", type=int, default=16)
ap.add_argument("--lora-alpha", type=int, default=32)
ap.add_argument("--eval-every", type=int, default=100)
ap.add_argument("--max-dev", type=int, default=None, help="cap the held-out training-object dev set (for speed)")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

model, tok = load_model()
examples = [json.loads(l) for l in open(f"data/sft/{args.data}/examples.jsonl")]
train = [tokenize_example(tok, e) for e in examples if not is_dev(e["key_id"])]
dev = {"train_cats": [tokenize_example(tok, e) for e in examples if is_dev(e["key_id"])][:args.max_dev],
       "val_cats": [tokenize_example(tok, json.loads(l)) for l in open(f"data/sft/{args.data}/dev_valcats.jsonl")]}
run = start_run(args.name, {**vars(args), "prompt_version": PROMPT_VERSION, "n_train": len(train),
                            "n_dev": {k: len(v) for k, v in dev.items()},
                            "model_record": json.load(open("models/model.json"))})
model = add_lora(model, r=args.lora_r, alpha=args.lora_alpha)
steps = train_sft(model, tok, train, dev, run, epochs=args.epochs, batch_size=args.batch_size,
                  micro_batch=args.micro_batch, lr=args.lr, seed=args.seed, eval_every=args.eval_every)
print(f"done: {steps} steps | adapter saved in {run / 'adapter'}")
