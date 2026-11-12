"""Reinforcement learning (GRPO-style): draw several sketches per prompt, reward each one by how recognisable
it is, and make above-average drawings more likely.

One step = rollout -> gate + reward -> advantages (reward minus group mean) -> one policy-gradient update.
"""
import json
import random
import re
import statistics
import time
from collections import defaultdict

import torch
from peft import PeftModel

from svgbudget.gate import check, ink_fraction
from svgbudget.judges import judge_reward, judge_top1
from svgbudget.model import load_model
from svgbudget.prompts import build_messages


def load_policy(adapter, device="cuda:0"):
    """Base model + SFT adapter, kept separate and trainable (RL keeps updating the adapter)."""
    base, tok = load_model(device=device)
    return PeftModel.from_pretrained(base, adapter, is_trainable=True), tok


@torch.no_grad()
def rollout(model, tok, tasks, group_size, max_new_tokens=1024, temperature=1.0, top_p=1.0, top_k=0,
            seed=0, device="cuda:0"):
    """group_size drawings per (category, budget) task. Keeps the EXACT prompt and generated token ids
    (the loss must be computed on the tokens that were actually sampled), including the stop token."""
    was_training = model.training
    model.eval()
    tok.padding_side = "left"
    texts = [tok.apply_chat_template(build_messages(c, n), tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False) for c, n in tasks]
    inputs = tok(texts, return_tensors="pt", padding=True).to(device)
    torch.manual_seed(seed)
    out = model.generate(**inputs, do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k,
                         max_new_tokens=max_new_tokens, num_return_sequences=group_size,
                         pad_token_id=tok.pad_token_id)
    model.train(was_training)
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    plen = inputs.input_ids.shape[1]
    records = []
    for row in range(out.shape[0]):
        t = row // group_size                                   # rows are grouped by task
        prompt_ids = inputs.input_ids[t][inputs.attention_mask[t].bool()].tolist()   # drop left padding
        gen = out[row, plen:].tolist()
        end = next((j for j, x in enumerate(gen) if x in eos), None)
        gen_ids = gen if end is None else gen[:end + 1]          # keep the stop token
        records.append({"task": t, "category": tasks[t][0], "n_shapes": tasks[t][1],
                        "prompt_ids": prompt_ids, "gen_ids": gen_ids, "finished": end is not None,
                        "text": tok.decode(gen_ids, skip_special_tokens=True)})
    return records


def group_advantages(rewards, groups, normalize_std=False, eps=1e-6):
    """Advantage = reward - the mean reward of its group (the drawings for the same prompt).
    Standard GRPO also divides by the group's std; we don't by default, because that blows up tiny,
    meaningless differences inside groups where every drawing failed."""
    by = defaultdict(list)
    for r, g in zip(rewards, groups):
        by[g].append(r)
    mean = {g: sum(v) / len(v) for g, v in by.items()}
    std = {g: (sum((x - mean[g]) ** 2 for x in v) / len(v)) ** 0.5 for g, v in by.items()}
    return [(r - mean[g]) / ((std[g] + eps) if normalize_std else 1.0) for r, g in zip(rewards, groups)]


def token_logprobs(model, chunk, pad_id, device):
    """log p of every generated token (right-padded prompt + answer), and a mask marking the answer tokens."""
    L = max(len(r["prompt_ids"]) + len(r["gen_ids"]) for r in chunk)
    ids = torch.full((len(chunk), L), pad_id, dtype=torch.long)
    attn = torch.zeros((len(chunk), L), dtype=torch.long)
    gen = torch.zeros((len(chunk), L), dtype=torch.float)
    for i, r in enumerate(chunk):
        seq = r["prompt_ids"] + r["gen_ids"]
        ids[i, :len(seq)] = torch.tensor(seq)
        attn[i, :len(seq)] = 1
        gen[i, len(r["prompt_ids"]):len(seq)] = 1
    ids, attn, gen = ids.to(device), attn.to(device), gen.to(device)
    logits = model(input_ids=ids, attention_mask=attn).logits[:, :-1].float()
    logp = torch.log_softmax(logits, dim=-1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return logp, gen[:, 1:]                                    # token t is predicted from tokens before t


def policy_loss_backward(model, recs, advantages, pad_id, micro_batch=4, device="cuda:0"):
    """loss = -(sum over answer tokens of advantage x log p(token)) / (all answer tokens in the batch).
    One update per batch of fresh rollouts, so no PPO ratio or clipping is needed.
    Calls backward() per micro-batch; skips drawings whose advantage is zero."""
    items = [(r, a) for r, a in zip(recs, advantages) if abs(a) > 1e-8]
    n_tok = sum(len(r["gen_ids"]) for r, _ in items)
    total = 0.0
    for m in range(0, len(items), micro_batch):
        chunk = items[m:m + micro_batch]
        logp, mask = token_logprobs(model, [r for r, _ in chunk], pad_id, device)
        adv = torch.tensor([a for _, a in chunk], device=device)[:, None]
        loss = -(logp * mask * adv).sum() / n_tok
        loss.backward()
        total += loss.item()
    return {"loss": total, "n_samples": len(items), "n_tokens": n_tok}


@torch.no_grad()
def sequence_logprobs(model, recs, pad_id, micro_batch=8, device="cuda:0"):
    """Total log p of each drawing under the current model (used by the 'does one step point the right way' test)."""
    was_training = model.training
    model.eval()
    out = []
    for m in range(0, len(recs), micro_batch):
        logp, mask = token_logprobs(model, recs[m:m + micro_batch], pad_id, device)
        out += (logp * mask).sum(dim=1).tolist()
    model.train(was_training)
    return out


def near_duplicate_strokes(svg, tol=2):
    """How many polylines nearly repeat an earlier one. Monitoring only, so it never raises."""
    try:
        strokes = []
        for m in re.findall(r'points="([^"]*)"', svg or ""):
            nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", m)]
            strokes.append(list(zip(nums[0::2], nums[1::2])))

        def near(a, b):
            return len(a) == len(b) and all(abs(x1 - x2) <= tol and abs(y1 - y2) <= tol
                                            for (x1, y1), (x2, y2) in zip(a, b))
        return sum(any(near(strokes[j], strokes[i]) for i in range(j)) for j in range(len(strokes)))
    except Exception:
        return 0


def score(recs, reward_judge):
    """Gate every drawing; reward the ones that pass (failures get 0)."""
    gates = [check(r["text"], r["n_shapes"]) for r in recs]
    ok = [i for i, g in enumerate(gates) if g["ok"]]
    rewards = [0.0] * len(recs)
    if ok:
        values = judge_reward(reward_judge, [gates[i]["image"] for i in ok], [recs[i]["category"] for i in ok])
        for i, v in zip(ok, values):
            rewards[i] = v
    return gates, rewards


def quick_eval(model, tok, tasks, reward_judge, check_judge, seed=12345, device="cuda:0"):
    """Fixed tasks, evaluation-style sampling and the same seed every time. Reports usable top-1 by the
    reward judge AND by an independent judge: if only the reward judge improves, that's reward hacking."""
    recs = rollout(model, tok, tasks, group_size=2, temperature=0.7, top_p=0.8, top_k=20, seed=seed, device=device)
    gates, rewards = score(recs, reward_judge)
    ok = [i for i, g in enumerate(gates) if g["ok"]]
    imgs, cats = [gates[i]["image"] for i in ok], [recs[i]["category"] for i in ok]
    hits_reward = sum(p == c for p, c in zip(judge_top1(reward_judge, imgs), cats)) if ok else 0
    hits_check = sum(p == c for p, c in zip(judge_top1(check_judge, imgs), cats)) if ok else 0
    n = len(recs)
    return {"gate": round(len(ok) / n, 3), "reward": round(sum(rewards) / n, 4),
            "usable_top1_reward_judge": round(hits_reward / n, 4),
            "usable_top1_check_judge": round(hits_check / n, 4)}


def train_rl(model, tok, reward_judge, check_judge, train_cats, eval_sets, run_dir, steps=200,
             prompts_per_step=16, group_size=8, lr=2e-6, warmup=10, max_grad_norm=1.0, budgets=(3, 6, 12),
             eval_every=25, save_every=50, seed=0, device="cuda:0"):
    """The RL loop. Logs every step to run_dir/rl_log.jsonl (reward plus reward-hacking monitors) and every
    drawing to run_dir/big/rollouts.jsonl."""
    rng = random.Random(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup))
    (run_dir / "big").mkdir(exist_ok=True)
    log = open(run_dir / "rl_log.jsonl", "a")
    roll = open(run_dir / "big" / "rollouts.jsonl", "a")

    def write(rec):
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(rec, flush=True)

    def evaluate(step):
        write({"step": step, **{f"eval_{k}": quick_eval(model, tok, t, reward_judge, check_judge, device=device)
                                for k, t in eval_sets.items()}})

    evaluate(0)
    for step in range(1, steps + 1):
        t0 = time.time()
        tasks = [(rng.choice(train_cats), budgets[k % len(budgets)]) for k in range(prompts_per_step)]
        recs = rollout(model, tok, tasks, group_size, seed=seed * 100_000 + step, device=device)
        gates, rewards = score(recs, reward_judge)
        adv = group_advantages(rewards, [r["task"] for r in recs])
        model.train()
        st = policy_loss_backward(model, recs, adv, tok.pad_token_id, device=device)
        gn = torch.nn.utils.clip_grad_norm_(params, max_grad_norm).item()
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)

        ok = [i for i, g in enumerate(gates) if g["ok"]]
        group_rewards = [[rewards[i] for i, r in enumerate(recs) if r["task"] == t] for t in range(len(tasks))]
        spread = [max(v) - min(v) for v in group_rewards]
        write({"step": step,
               "reward": round(statistics.mean(rewards), 4),
               **{f"reward_N{n}": round(statistics.mean([rewards[i] for i, r in enumerate(recs) if r["n_shapes"] == n]), 4)
                  for n in budgets},
               "gate": round(len(ok) / len(recs), 3),
               "finished": round(sum(r["finished"] for r in recs) / len(recs), 3),
               "useful_groups": round(sum(s > 0.1 for s in spread) / len(spread), 3),
               "gen_tokens_median": statistics.median(len(r["gen_ids"]) for r in recs),
               "shapes_median": statistics.median(gates[i]["n_shapes"] for i in ok) if ok else None,
               "ink_1_2pct": sum(0.01 <= ink_fraction(gates[i]["image"]) < 0.02 for i in ok),
               "dup_drawings": round(sum(near_duplicate_strokes(r["text"]) > 0 for r in recs) / len(recs), 3),
               "loss": round(st["loss"], 6), "grad_norm": round(gn, 4), "lr": sched.get_last_lr()[0],
               "seconds": round(time.time() - t0, 1)})
        for r, g, v in zip(recs, gates, rewards):
            roll.write(json.dumps({"step": step, "category": r["category"], "n_shapes": r["n_shapes"],
                                   "reward": round(v, 4), "reason": g["reason"], "text": r["text"]}) + "\n")
        roll.flush()
        if step % eval_every == 0:
            evaluate(step)
        if step % save_every == 0:
            model.save_pretrained(run_dir / f"adapter_step{step}")
    if steps % eval_every:
        evaluate(steps)
    model.save_pretrained(run_dir / "adapter")
    log.close()
    roll.close()
