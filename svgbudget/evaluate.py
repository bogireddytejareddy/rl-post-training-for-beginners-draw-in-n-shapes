"""One evaluation pipeline for every model: generate -> gate -> judges -> summary."""
import statistics
from collections import Counter

import torch

from svgbudget.gate import check
from svgbudget.judges import embed_images, load_judge, rank_of_truth, top_k
from svgbudget.model import generate_batch
from svgbudget.prompts import build_messages

# Each judge's top-1 accuracy on 660 human sketches of the validation objects. Scores are reported next to these.
HUMAN_CEILING_TOP1 = {"clip": 0.336, "siglip2": 0.432}


def generate_all(model, tok, tasks, cfg, batch_prompts=24):
    """Sample drawings for every (category, budget) task. The seed is the batch number, so runs repeat exactly."""
    out = []
    n_batches = -(-len(tasks) // batch_prompts)
    for b in range(0, len(tasks), batch_prompts):
        chunk = tasks[b:b + batch_prompts]
        res = generate_batch(model, tok, [build_messages(c, n) for c, n in chunk], seed=b // batch_prompts, **cfg)
        for (c, n), samples in zip(chunk, res):
            for k, s in enumerate(samples):
                out.append({"category": c, "n_shapes": n, "sample": k, **s})
        print(f"  batch {b // batch_prompts + 1}/{n_batches}", flush=True)
    return out


def gate_all(records):
    """Run the gate on every record (adds ok / reason / detail / shapes_used). Returns the images (None = failed)."""
    images = []
    for r in records:
        g = check(r["text"], r["n_shapes"])
        r.update(ok=g["ok"], reason=g["reason"], detail=g["detail"], shapes_used=g["n_shapes"])
        images.append(g["image"])
    return images


def judge_all(records, images, judge_paths, candidates):
    """Blind identification by each judge, for drawings that passed the gate (adds <judge>_top5, <judge>_rank)."""
    ok = [i for i, r in enumerate(records) if r["ok"]]
    for name, path in judge_paths.items():
        for r in records:
            r[f"{name}_top5"], r[f"{name}_rank"] = None, None
        judge = load_judge(path, candidates)
        iv = embed_images(judge["model"], judge["proc"], [images[i] for i in ok])
        preds = top_k(iv, judge["text_vecs"])
        ranks = rank_of_truth(iv, judge["text_vecs"], [judge["index"][records[i]["category"]] for i in ok])
        for i, p, rk in zip(ok, preds, ranks):
            records[i][f"{name}_top5"] = [candidates[j] for j in p]
            records[i][f"{name}_rank"] = rk
        del judge, iv
        torch.cuda.empty_cache()


def summarize(records, judge_names, n_candidates):
    """Numbers per budget and overall.
    usable = passed the gate AND the judge's first choice is right, as a share of ALL outputs."""
    groups = {f"N={n}": [r for r in records if r["n_shapes"] == n] for n in sorted({r["n_shapes"] for r in records})}
    groups["all"] = records
    out = {}
    for g, rs in groups.items():
        n = len(rs)
        s = {"n": n, "gate_pass": sum(r["ok"] for r in rs) / n,
             "truncated": sum(not r["finished"] for r in rs),
             "reasons": dict(Counter(r["reason"] for r in rs).most_common())}
        for j in judge_names:
            u1 = sum(r["ok"] and r[f"{j}_top5"][0] == r["category"] for r in rs) / n
            s[j] = {"usable_top1": u1,
                    "usable_top5": sum(r["ok"] and r["category"] in r[f"{j}_top5"] for r in rs) / n,
                    "usable_top1_vs_human": u1 / HUMAN_CEILING_TOP1[j],
                    # failed drawings count as ranked last
                    "median_rank": statistics.median(r[f"{j}_rank"] if r["ok"] else n_candidates for r in rs)}
        out[g] = s
    return out
