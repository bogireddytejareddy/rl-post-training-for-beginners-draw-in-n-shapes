"""Step 1: download everything, pinned to exact versions.

  - Qwen3-1.7B (the model we train) and the two judges (CLIP ViT-L/14, SigLIP 2)
  - Quick, Draw! sketches: 1000 per training object, 10 per validation object

    python scripts/01_download.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from svgbudget.download import download_model, download_sketches

splits = json.load(open("prompts/splits.json"))
Path("models").mkdir(exist_ok=True)

model = download_model("Qwen/Qwen3-1.7B")
json.dump(model, open("models/model.json", "w"), indent=2)
judges = {"clip": download_model("openai/clip-vit-large-patch14")["local_path"],
          "siglip2": download_model("google/siglip2-so400m-patch14-384")["local_path"]}
json.dump(judges, open("models/judges.json", "w"), indent=2)
print("model:", model["revision"], "| judges:", list(judges))

out_dir = Path("data/quickdraw")
out_dir.mkdir(parents=True, exist_ok=True)
for split, k in [("train", 1000), ("val", 10)]:
    versions = []
    with open(out_dir / f"{split}.jsonl", "w") as f:
        for c in splits[split]:
            d = download_sketches(c, k=k)                   # only sketches Google's own classifier recognised
            for s in d["sketches"]:
                f.write(json.dumps({"category": c, **s}) + "\n")
            versions.append({k2: d[k2] for k2 in ("category", "url", "ETag", "Last-Modified")})
    json.dump(versions, open(out_dir / f"{split}_versions.json", "w"), indent=1)
    print(f"{split}: {len(splits[split])} objects x {k} sketches")
