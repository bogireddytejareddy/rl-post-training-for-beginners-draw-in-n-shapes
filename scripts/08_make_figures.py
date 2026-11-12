"""Step 8: every chart in the README, from the numbers in results/.

    python scripts/08_make_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("results")
F = Path("figures")
F.mkdir(exist_ok=True)
ev = json.load(open(R / "evaluations.json"))
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, GREY, GREEN, RED = "#3b6fb6", "#e08a2e", "#9a9a9a", "#4a9a5b", "#c0504d"


def top1(key, judge="siglip2"):
    return 100 * ev[key]["summary"]["all"][judge]["usable_top1"]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(F / name, dpi=150)
    plt.close(fig)
    print("saved", F / name)


# 1. the headline: data scaling, seen vs unseen objects
x = [100, 300, 1000]
fig, ax = plt.subplots(figsize=(7.5, 4.6))
for split, color in [("seen", BLUE), ("unseen", ORANGE)]:
    keys = [f"sft_v2a_{split}", f"sft_v3_300_{split}", f"sft_v3_1000_{split}"]
    ax.plot(x, [top1(k) for k in keys], "o-", color=color, lw=2.2, label=f"{split} objects, SFT (SigLIP 2)")
    ax.plot(x, [top1(k, "clip") for k in keys], "o--", color=color, lw=1.4, alpha=0.7, label=f"{split} objects, SFT (CLIP)")
    ax.plot([1000], [top1(f"rl_v1_{split}")], "*", color=color, ms=16, label=f"{split} objects, SFT + RL (SigLIP 2)")
    ax.axhline(top1(f"baseline_{split}"), color=color, ls=":", lw=1)
ax.axhline(44, color=GREY, ls="-.", lw=1)
ax.text(105, 45, "human sketches of the same objects (~44%)", color=GREY, fontsize=9)
ax.text(105, 2.5, "dotted: untrained baseline", color=GREY, fontsize=9)
ax.set_xscale("log")
ax.set_xticks(x, ["100", "300", "1000"])
ax.set_xlabel("human sketches per training object (SFT data)")
ax.set_ylabel("usable drawings, top-1 (%)")
ax.set_ylim(0, 50)
ax.set_title("More data teaches the objects it has seen, not new ones")
ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 0.84), frameon=False)
save(fig, "fig_scaling.png")

# 2. every stage, both judges, both splits
stages = [("baseline", "Baseline"), ("sft_v1", "SFT v1"), ("sft_v2a", "SFT v2a\n100/obj"), ("sft_v2b", "SFT v2b"),
          ("sft_v3_300", "SFT\n300/obj"), ("sft_v3_1000", "SFT\n1000/obj"), ("rl_v1", "SFT 1000\n+ RL")]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
for ax, split in zip(axes, ["seen", "unseen"]):
    present = [(k, l) for k, l in stages if f"{k}_{split}" in ev]
    for j, (judge, color) in enumerate([("siglip2", BLUE), ("clip", ORANGE)]):
        vals = [top1(f"{k}_{split}", judge) for k, _ in present]
        bars = ax.bar([i + (j - 0.5) * 0.38 for i in range(len(present))], vals, 0.38, color=color,
                      label="SigLIP 2 (gives the RL reward)" if judge == "siglip2" else "CLIP (independent check)")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.1f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(present)), [l for _, l in present], fontsize=9)
    ax.set_title(f"{split} objects ({'training' if split == 'seen' else 'validation'} categories)")
axes[0].set_ylabel("usable drawings, top-1 (%)")
axes[0].legend(frameon=False, fontsize=9)
save(fig, "fig_stages.png")

# 3. why drawings fail (unseen objects)
order = [("baseline", "Baseline"), ("sft_v1", "SFT v1"), ("sft_v2a", "v2a"), ("sft_v2b", "v2b"),
         ("sft_v3_300", "300/obj"), ("sft_v3_1000", "1000/obj"), ("rl_v1", "+ RL")]
parts = [("ok", "valid drawing", GREEN), ("no_svg", "never finished (loop)", RED),
         ("over_budget", "too many shapes", ORANGE), ("blank", "blank canvas", GREY)]
fig, ax = plt.subplots(figsize=(9, 4.2))
bottom = [0] * len(order)
for reason, label, color in parts:
    vals = []
    for k, _ in order:
        s = ev[f"{k}_unseen"]["summary"]["all"]
        vals.append(100 * s["reasons"].get(reason, 0) / s["n"])
    ax.bar(range(len(order)), vals, bottom=bottom, color=color, label=label)
    bottom = [b + v for b, v in zip(bottom, vals)]
ax.bar(range(len(order)), [100 - b for b in bottom], bottom=bottom, color="#333333", label="other")
ax.set_xticks(range(len(order)), [l for _, l in order])
ax.set_ylabel("share of all 792 outputs (%)")
ax.set_title("Gate results on unseen objects: what happens to every output")
ax.legend(frameon=False, fontsize=9, bbox_to_anchor=(1.0, 1.0), loc="upper left")
save(fig, "fig_failures.png")

# 4. SFT dev loss (same held-out unseen-object examples for these three runs)
fig, ax = plt.subplots(figsize=(7.5, 4.3))
for name, label, color in [("v2a", "100/object", GREY), ("v3_300", "300/object", ORANGE), ("v3_1000", "1000/object", BLUE)]:
    rows = [json.loads(l) for l in open(R / "sft_logs" / f"sft_{name}.jsonl")]
    dev = [r for r in rows if "dev_val_cats" in r]
    xs = [r["step"] * 16 / 1000 for r in dev]
    ax.plot(xs, [r["dev_val_cats"] for r in dev], "-", color=color, lw=2, label=f"{label}: unseen objects")
    ax.plot(xs, [r["dev_train_cats"] for r in dev], "--", color=color, lw=1.2, label=f"{label}: held-out sketches of seen objects")
ax.set_ylim(0.62, 0.85)
ax.set_xlabel("training examples seen (thousands)")
ax.set_ylabel("held-out loss per token")
ax.set_title("SFT: the gap between seen and unseen objects grows with data")
ax.legend(frameon=False, fontsize=8)
save(fig, "fig_sft_loss.png")

# 5. RL training
rows = [json.loads(l) for l in open(R / "rl_log.jsonl")]
steps = [r for r in rows if "reward" in r]
evals = [r for r in rows if "eval_train" in r]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
s = [r["step"] for r in steps]
rw = [r["reward"] for r in steps]
avg = [sum(rw[max(0, i - 24):i + 1]) / len(rw[max(0, i - 24):i + 1]) for i in range(len(rw))]
axes[0].plot(s, rw, color=BLUE, alpha=0.25, lw=1, label="each step (128 drawings)")
axes[0].plot(s, avg, color=BLUE, lw=2.4, label="25-step average")
axes[0].set_xlabel("RL step")
axes[0].set_ylabel("mean reward")
axes[0].set_title("Reward during RL (temperature 1.0)")
axes[0].legend(frameon=False, fontsize=9)
es = [r["step"] for r in evals]
axes[1].plot(es, [100 * r["eval_train"]["usable_top1_reward_judge"] for r in evals], "o-", color=BLUE, label="SigLIP 2 (reward judge)")
axes[1].plot(es, [100 * r["eval_train"]["usable_top1_check_judge"] for r in evals], "o-", color=ORANGE, label="CLIP (independent)")
axes[1].plot(es, [100 * r["eval_val"]["usable_top1_reward_judge"] for r in evals], "s:", color=BLUE, alpha=0.6, label="unseen, SigLIP 2")
axes[1].set_xlabel("RL step")
axes[1].set_ylabel("usable top-1 (%), 132 drawings")
axes[1].set_title("Quick evaluations (noisy: about +/-4 points)")
axes[1].legend(frameon=False, fontsize=8)
axes[2].plot(s, [100 * r["gate"] for r in steps], color=GREEN, lw=1.2, label="gate pass (%)")
axes[2].plot(s, [100 * r["dup_drawings"] for r in steps], color=RED, lw=1.2, label="drawings with copied strokes (%)")
axes[2].plot(s, [r["ink_1_2pct"] for r in steps], color=GREY, lw=1.2, label="near-blank drawings (count)")
axes[2].set_ylim(-2, 105)
axes[2].set_xlabel("RL step")
axes[2].set_title("Reward-hacking monitors stayed quiet")
axes[2].legend(frameon=False, fontsize=8)
save(fig, "fig_rl.png")

# 6. judge calibration
cal = json.load(open(R / "judge_calibration.json"))
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
same = cal["all_judges_same_198_sketches_top1_top5"]
names = ["siglip2", "clip", "gpt-5-mini", "gpt-4o"]
axes[0].bar(range(4), [100 * same[n][0] for n in names], color=[BLUE, ORANGE, GREY, GREY])
for i, n in enumerate(names):
    axes[0].text(i, 100 * same[n][0] + 1, f"{100 * same[n][0]:.1f}", ha="center")
axes[0].axhline(0.3, color=RED, ls=":")
axes[0].set_xticks(range(4), ["SigLIP 2", "CLIP", "GPT-5-mini", "GPT-4o"])
axes[0].set_ylabel("top-1 on 198 human sketches (%)")
axes[0].set_title("Judges on human sketches (chance = 0.3%)")
cb = cal["human_sketch_ceiling_by_strokes_top1"]
xx = range(len(cb["strokes"]))
axes[1].bar([i - 0.2 for i in xx], [100 * v for v in cb["siglip2"]], 0.4, color=BLUE, label="SigLIP 2")
axes[1].bar([i + 0.2 for i in xx], [100 * v for v in cb["clip"]], 0.4, color=ORANGE, label="CLIP")
axes[1].set_xticks(list(xx), [f"{s} strokes\n(n={n})" for s, n in zip(cb["strokes"], cb["n"])])
axes[1].set_ylabel("top-1 on human sketches (%)")
axes[1].set_title("Few-stroke human sketches are still recognised")
axes[1].legend(frameon=False)
save(fig, "fig_judges.png")
