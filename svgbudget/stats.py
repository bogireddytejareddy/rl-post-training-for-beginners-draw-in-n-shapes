"""Small helpers for summarising lists of numbers."""
import numpy as np


def length_stats(values):
    """Mean and percentiles. We use percentiles because lengths here are heavily skewed."""
    a = np.asarray(values)
    out = {"n": int(a.size), "mean": round(float(a.mean()), 3)}
    for q in (10, 50, 90, 99):
        out[f"p{q}"] = float(np.percentile(a, q))
    out["max"] = a.max().item()
    return out


def token_lengths(texts, tokenizer, batch_size=1000):
    """Number of tokens in each text (the text only, no special tokens)."""
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i:i + batch_size], add_special_tokens=False)
        out.extend(len(ids) for ids in enc["input_ids"])
    return out
