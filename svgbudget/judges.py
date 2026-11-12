"""Layer 2, "the judge": can a vision model tell which object a drawing shows, without seeing the prompt?

A judge (CLIP or SigLIP 2) turns the image and each of the 331 category names into vectors; its answer
is the category whose vector is closest. The same judge also gives the RL reward: the probability it
puts on the TRUE category.
"""
import json

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor

TEXT_TEMPLATE = "a simple drawing of a {}"
JUDGES_RECORD = "models/judges.json"   # {"clip": path, "siglip2": path}, written by scripts/01_download.py


def judge_paths(record_path=JUDGES_RECORD):
    with open(record_path) as f:
        return json.load(f)


def _features(out):
    """Some transformers versions return a tensor, others an output object."""
    return out if torch.is_tensor(out) else out.pooler_output


@torch.inference_mode()
def embed_texts(model, proc, texts, device="cuda:0"):
    """Unit-length text vectors. SigLIP models must be padded to 64 tokens, as in their training."""
    if "siglip" in model.config.model_type:
        pad = dict(padding="max_length", max_length=64)
    else:
        pad = dict(padding=True)
    inp = proc(text=texts, return_tensors="pt", truncation=True, **pad).to(device)
    return F.normalize(_features(model.get_text_features(**inp)), dim=-1)


@torch.inference_mode()
def embed_images(model, proc, images, device="cuda:0", batch_size=64):
    """Unit-length image vectors, in batches."""
    out = []
    for i in range(0, len(images), batch_size):
        inp = proc(images=images[i:i + batch_size], return_tensors="pt").to(device)
        out.append(F.normalize(_features(model.get_image_features(**inp)), dim=-1))
    return torch.cat(out)


def top_k(image_vecs, text_vecs, k=5):
    """Indices of the k closest categories for each image."""
    return (image_vecs @ text_vecs.T).topk(k, dim=-1).indices.cpu().tolist()


def rank_of_truth(image_vecs, text_vecs, truth):
    """Where the true category ranks among all candidates (1 = the judge's first choice)."""
    sims = image_vecs @ text_vecs.T
    t = sims[torch.arange(len(truth), device=sims.device), torch.tensor(truth, device=sims.device)]
    return ((sims > t[:, None]).sum(dim=1) + 1).cpu().tolist()


def load_judge(path, candidates, device="cuda:0"):
    """Everything a judge needs, in a plain dict: the model, its processor, and the 331 category vectors."""
    model = AutoModel.from_pretrained(path).to(device).eval()
    proc = AutoProcessor.from_pretrained(path)
    return {"model": model, "proc": proc, "device": device,
            "candidates": list(candidates),
            "index": {c: i for i, c in enumerate(candidates)},
            "text_vecs": embed_texts(model, proc, [TEXT_TEMPLATE.format(c) for c in candidates], device),
            "scale": model.logit_scale.exp().item()}         # the judge's own sharpness


@torch.inference_mode()
def judge_top1(judge, images):
    """The judge's first-choice category for each image."""
    iv = embed_images(judge["model"], judge["proc"], images, judge["device"])
    best = (iv @ judge["text_vecs"].T).argmax(dim=-1).tolist()
    return [judge["candidates"][b] for b in best]


@torch.inference_mode()
def judge_reward(judge, images, categories):
    """RL reward: the judge's probability (softmax over all 331 categories) for each drawing's TRUE category."""
    iv = embed_images(judge["model"], judge["proc"], images, judge["device"])
    probs = (judge["scale"] * iv @ judge["text_vecs"].T).softmax(dim=-1)
    truth = torch.tensor([judge["index"][c] for c in categories], device=probs.device)
    return probs[torch.arange(len(images), device=probs.device), truth].tolist()
