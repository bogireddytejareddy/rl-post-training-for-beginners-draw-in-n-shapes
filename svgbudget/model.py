"""Loading Qwen3-1.7B (optionally with a LoRA adapter) and sampling drawings from it."""
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

hf_logging.disable_progress_bar()

MODEL_RECORD = "models/model.json"     # written by scripts/01_download.py


def load_model(adapter=None, record_path=MODEL_RECORD, dtype=torch.bfloat16, device="cuda:0"):
    """The pinned base model on one GPU, in eval mode. A LoRA adapter, if given, is merged into the weights
    (faster generation; only for evaluation - RL keeps the adapter separate, see rl.load_policy)."""
    with open(record_path) as f:
        path = json.load(f)["local_path"]
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(device)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    return model.eval(), tok


def generate_batch(model, tok, messages_list, n_samples=4, max_new_tokens=1024, temperature=0.7, top_p=0.8,
                   top_k=20, enable_thinking=False, seed=0):
    """Sample n_samples answers for each prompt. Returns one list of {text, n_tokens, finished} per prompt."""
    tok.padding_side = "left"          # batching a left-to-right model needs left padding
    texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True,
                                     enable_thinking=enable_thinking) for m in messages_list]
    inputs = tok(texts, return_tensors="pt", padding=True).to(model.device)
    torch.manual_seed(seed)
    with torch.inference_mode():
        out = model.generate(**inputs, do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k,
                             min_p=0.0, max_new_tokens=max_new_tokens, num_return_sequences=n_samples,
                             pad_token_id=tok.pad_token_id)
    new = out[:, inputs.input_ids.shape[1]:].tolist()
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    results = []
    for i in range(len(messages_list)):
        samples = []
        for row in new[i * n_samples:(i + 1) * n_samples]:      # rows are grouped by prompt
            end = next((j for j, t in enumerate(row) if t in eos), None)
            ids = row if end is None else row[:end]
            samples.append({"text": tok.decode(ids, skip_special_tokens=True),
                            "n_tokens": len(ids), "finished": end is not None})
        results.append(samples)
    return results
