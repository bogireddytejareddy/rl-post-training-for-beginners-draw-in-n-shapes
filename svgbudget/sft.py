"""Supervised fine-tuning (SFT) with LoRA: teach the model to write the human sketch for each prompt."""
import hashlib
import json
import math
import random
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import get_cosine_schedule_with_warmup

from svgbudget.prompts import build_messages

IGNORE = -100        # label value the loss skips
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def tokenize_example(tok, ex, enable_thinking=False):
    """Prompt (built exactly as at generation time) + target SVG + <|im_end|>.
    Prompt tokens get label IGNORE, so the model only learns to produce the answer (and to stop)."""
    prompt = tok.apply_chat_template(build_messages(ex["category"], ex["n_shapes"]), tokenize=False,
                                     add_generation_prompt=True, enable_thinking=enable_thinking)
    p_ids = tok(prompt, add_special_tokens=False).input_ids
    t_ids = tok(ex["target"] + "<|im_end|>", add_special_tokens=False).input_ids
    return {"input_ids": p_ids + t_ids, "labels": [IGNORE] * len(p_ids) + t_ids}


def collate(batch, pad_id):
    """Right-pad a batch. Padding gets label IGNORE and attention 0: never trained on, never looked at."""
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), n), IGNORE, dtype=torch.long)
    attn = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        ids[i, :L] = torch.tensor(b["input_ids"])
        labels[i, :L] = torch.tensor(b["labels"])
        attn[i, :L] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": attn}


def add_lora(model, r=16, alpha=32, dropout=0.05):
    """Small trainable adapters (about 1% of the weights) on every attention and feed-forward projection."""
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=LORA_TARGETS, task_type="CAUSAL_LM")
    return get_peft_model(model, cfg)


def n_trained(batch):
    """Tokens the loss averages over (the model shifts labels by one internally)."""
    return (batch["labels"][:, 1:] != IGNORE).sum().item()


@torch.inference_mode()
def eval_loss(model, tok, tokenized, batch_size=16, device="cuda:0"):
    """Average loss per trained token over held-out examples (dropout off while measuring)."""
    was_training = model.training
    model.eval()
    total, n_tok = 0.0, 0
    for i in range(0, len(tokenized), batch_size):
        b = {k: v.to(device) for k, v in collate(tokenized[i:i + batch_size], tok.pad_token_id).items()}
        n = n_trained(b)
        total += model(**b).loss.item() * n
        n_tok += n
    model.train(was_training)
    return total / n_tok


def is_dev(key_id, every=20):
    """Deterministic ~5% held-out split. We hash first because raw Quick, Draw! IDs are all multiples of 4."""
    return int(hashlib.md5(key_id.encode()).hexdigest(), 16) % every == 0


def train_sft(model, tok, train_tok, dev_sets, run_dir, epochs=2, batch_size=16, micro_batch=4, lr=2e-4,
              warmup_frac=0.03, max_grad_norm=1.0, seed=0, log_every=10, eval_every=100, device="cuda:0"):
    """The SFT loop. Each batch of 16 is split into micro-batches of 4 to fit in memory; each micro-batch is
    weighted by its share of the batch's tokens, which gives exactly the full batch's gradient.
    Logs to run_dir/train_log.jsonl; saves the adapter after each epoch and at the end."""
    rng = random.Random(seed)
    torch.manual_seed(seed)
    total = math.ceil(len(train_tok) / batch_size) * epochs
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, int(warmup_frac * total), total)
    log = open(run_dir / "train_log.jsonl", "a")

    def write(rec):
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(rec, flush=True)

    def evaluate(step):
        write({"step": step, **{f"dev_{k}": round(eval_loss(model, tok, v, device=device), 4)
                                for k, v in dev_sets.items()}})

    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    step, t0, recent = 0, time.time(), []
    evaluate(0)
    for epoch in range(epochs):
        order = list(range(len(train_tok)))
        rng.shuffle(order)
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            micro = [collate([train_tok[j] for j in idx[m:m + micro_batch]], tok.pad_token_id)
                     for m in range(0, len(idx), micro_batch)]
            n_total = sum(n_trained(mb) for mb in micro)
            batch_loss = 0.0
            for mb in micro:
                mb = {k: v.to(device) for k, v in mb.items()}
                w = n_trained(mb) / n_total
                loss = model(**mb).loss
                (loss * w).backward()
                batch_loss += loss.item() * w
            gn = torch.nn.utils.clip_grad_norm_(params, max_grad_norm).item()
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            recent.append(batch_loss)
            if step % log_every == 0:
                write({"step": step, "epoch": epoch, "train_loss": round(sum(recent) / len(recent), 4),
                       "lr": sched.get_last_lr()[0], "grad_norm": round(gn, 3), "elapsed_s": round(time.time() - t0),
                       "peak_mem_gb": round(torch.cuda.max_memory_allocated(device) / 1e9, 1)})
                recent = []
            if step % eval_every == 0:
                evaluate(step)
        model.save_pretrained(run_dir / f"adapter_epoch{epoch + 1}")
    if step % eval_every:
        evaluate(step)
    model.save_pretrained(run_dir / "adapter")
    log.close()
    return step
