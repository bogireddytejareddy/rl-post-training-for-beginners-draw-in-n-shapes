"""Turning Quick, Draw! sketches into SFT training examples."""
import random
import re
from collections import Counter, defaultdict


def rdp(points, eps):
    """Ramer-Douglas-Peucker line simplification: drop points closer than eps to the line between kept
    neighbours. Both ends are always kept, so closed shapes stay closed."""
    if len(points) < 3:
        return list(points)
    keep = [False] * len(points)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        a, b = stack.pop()
        (x1, y1), (x2, y2) = points[a], points[b]
        dx, dy = x2 - x1, y2 - y1
        norm = (dx * dx + dy * dy) ** 0.5
        best, best_d = None, eps
        for i in range(a + 1, b):
            x0, y0 = points[i]
            if norm:
                d = abs(dy * (x0 - x1) - dx * (y0 - y1)) / norm
            else:
                d = ((x0 - x1) ** 2 + (y0 - y1) ** 2) ** 0.5
            if d > best_d:
                best, best_d = i, d
        if best is not None:
            keep[best] = True
            stack += [(a, best), (best, b)]
    return [p for p, k in zip(points, keep) if k]


def sketch_to_compact_svg(drawing, size=100, margin=5, stroke_width=2, eps=0):
    """Quick, Draw! strokes -> the SVG text the model learns to write.
    0-100 canvas, whole-number coordinates (each digit is a token), one <polyline> per stroke,
    shared style written once on a <g>. eps > 0 simplifies every stroke with rdp()."""
    xs = [x for stroke in drawing for x in stroke[0]]
    ys = [y for stroke in drawing for y in stroke[1]]
    w, h = max(xs) - min(xs), max(ys) - min(ys)
    scale = (size - 2 * margin) / max(w, h, 1)
    ox = (size - w * scale) / 2 - min(xs) * scale
    oy = (size - h * scale) / 2 - min(ys) * scale
    lines = []
    for sx, sy in drawing:
        pts = []
        for x, y in zip(sx, sy):
            p = (round(x * scale + ox), round(y * scale + oy))
            if not pts or p != pts[-1]:          # rounding can make neighbours identical; keep one
                pts.append(p)
        if eps:
            pts = rdp(pts, eps)
        if len(pts) == 1:
            pts = pts * 2                         # a dot needs two points to render
        lines.append('<polyline points="' + " ".join(f"{x},{y}" for x, y in pts) + '"/>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}">\n'
            f'<g fill="none" stroke="black" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round">\n'
            + "\n".join(lines) + "\n</g>\n</svg>")


def build_sft_examples(rows, to_svg, n_tokens, passes_gate, budgets=(3, 6, 12), per_budget=None,
                       token_cap=900, seed=0, max_points=None):
    """Pair each sketch with ONE random budget it fits (so the model learns "at most N", not "exactly N"),
    then take the same number of examples per budget, spread round-robin over categories.

    to_svg(drawing) -> svg, n_tokens(svg) -> int, passes_gate(svg, n) -> bool are passed in.
    per_budget=None means: as many as the smallest budget pool allows."""
    rng = random.Random(seed)
    dropped = Counter()
    pools = defaultdict(list)                       # budget -> candidate examples
    for r in rows:
        strokes = len(r["drawing"])
        valid = [n for n in budgets if n >= strokes]
        if not valid:
            dropped["too_many_strokes"] += 1
            continue
        svg = to_svg(r["drawing"])
        if not passes_gate(svg, strokes):
            dropped["gate"] += 1
            continue
        if max_points and max(len(m.split()) for m in re.findall(r'points="([^"]*)"', svg)) > max_points:
            dropped["over_max_points"] += 1
            continue
        t = n_tokens(svg)
        if t > token_cap:
            dropped["over_token_cap"] += 1
            continue
        n = rng.choice(valid)
        pools[n].append({"category": r["category"], "key_id": r["key_id"], "n_shapes": n,
                         "strokes": strokes, "tokens": t, "target": svg})
    if per_budget is None:
        per_budget = min(len(pools[n]) for n in budgets)
    chosen = []
    for n in budgets:
        by_cat = defaultdict(list)
        for ex in pools[n]:
            by_cat[ex["category"]].append(ex)
        cats = sorted(by_cat)
        for c in cats:
            rng.shuffle(by_cat[c])
        rng.shuffle(cats)
        picked, depth = [], 0
        while len(picked) < per_budget and any(len(by_cat[c]) > depth for c in cats):
            for c in cats:                          # one example per category per pass
                if len(by_cat[c]) > depth and len(picked) < per_budget:
                    picked.append(by_cat[c][depth])
            depth += 1
        chosen += picked
    rng.shuffle(chosen)
    stats = {"per_budget": per_budget, "pool_sizes": {n: len(pools[n]) for n in budgets},
             "dropped": dict(dropped), "chosen_per_budget": dict(Counter(ex["n_shapes"] for ex in chosen))}
    return chosen, stats
