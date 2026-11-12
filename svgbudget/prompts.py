"""The task: which objects go in which split, and the exact prompt the model sees."""
import random

PROMPT_VERSION = "v1-minimal"

PROMPT_TEMPLATE = """Draw the following subject as an SVG image: {subject}

Rules:
- Use at most {n} shapes. Shape elements are <circle>, <ellipse>, <rect>, <line>, <polyline>, <polygon> and <path>. Every subpath of a <path> (each M or m command) counts as one shape.
- Do not use <text>, <image>, <use>, <style>, patterns, filters, masks or clip paths.

Reply with only the SVG code."""


def make_splits(categories, exclude, fractions=(0.6, 0.2, 0.2), seed=0):
    """Split CATEGORIES (not prompts) into train/val/test, so val and test objects are never seen in training."""
    unknown = set(exclude) - set(categories)
    if unknown:
        raise ValueError(f"exclusions not in the category list (typo?): {sorted(unknown)}")
    kept = sorted(c for c in categories if c not in set(exclude))
    random.Random(seed).shuffle(kept)
    n_train = round(fractions[0] * len(kept))
    n_val = round(fractions[1] * len(kept))
    return {"train": sorted(kept[:n_train]),
            "val": sorted(kept[n_train:n_train + n_val]),
            "test": sorted(kept[n_train + n_val:])}


def build_messages(subject, n_shapes):
    """The chat messages for one drawing task."""
    return [{"role": "user", "content": PROMPT_TEMPLATE.format(subject=subject, n=n_shapes)}]


def all_candidates(splits):
    """Every category a judge can choose from (all 331, from every split)."""
    return sorted(splits["train"] + splits["val"] + splits["test"])
