"""Step 3: can the judges be trusted? Before scoring any model, test each judge on
  - human sketches of the validation objects (positive control: should be recognised)
  - random coloured shapes with random labels (negative control: should be at chance, 0.3%)
  - a blank canvas (which objects does the judge guess when there is nothing to see?)

    python scripts/03_calibrate_judges.py
"""
import json
import random
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from svgbudget.judges import embed_images, judge_paths, load_judge, top_k
from svgbudget.prompts import all_candidates
from svgbudget.render import render_svg, sketch_to_svg


def random_drawing(rng, n):
    shapes = []
    for _ in range(n):
        color = "#%06x" % rng.randrange(1 << 24)
        if rng.random() < 0.5:
            shapes.append(f'<circle cx="{rng.randint(10, 90)}" cy="{rng.randint(10, 90)}" r="{rng.randint(5, 30)}" fill="{color}"/>')
        else:
            shapes.append(f'<rect x="{rng.randint(0, 70)}" y="{rng.randint(0, 70)}" width="{rng.randint(5, 30)}" '
                          f'height="{rng.randint(5, 30)}" fill="{color}"/>')
    return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">' + "".join(shapes) + "</svg>"


splits = json.load(open("prompts/splits.json"))
candidates = all_candidates(splits)
sketches = [json.loads(l) for l in open("data/quickdraw/val.jsonl")]
rng = random.Random(0)
random_imgs = [render_svg(random_drawing(rng, n)) for n in [3, 6, 12] * 100]
random_labels = [rng.choice(splits["val"]) for _ in range(300)]
groups = {"human sketches": ([render_svg(sketch_to_svg(s["drawing"])) for s in sketches], [s["category"] for s in sketches]),
          "random drawings": (random_imgs, random_labels)}

results = {}
for name, path in judge_paths().items():
    judge = load_judge(path, candidates)
    results[name] = {}
    for group, (imgs, labels) in groups.items():
        preds = top_k(embed_images(judge["model"], judge["proc"], imgs), judge["text_vecs"])
        truth = [judge["index"][c] for c in labels]
        results[name][group] = {"n": len(imgs),
                                "top1": sum(p[0] == t for p, t in zip(preds, truth)) / len(imgs),
                                "top5": sum(t in p for p, t in zip(preds, truth)) / len(imgs)}
    blank = top_k(embed_images(judge["model"], judge["proc"], [Image.new("RGB", (256, 256), "white")]), judge["text_vecs"])[0]
    results[name]["blank canvas guesses"] = [candidates[j] for j in blank]
    print(name, json.dumps(results[name], indent=1))
Path("results").mkdir(exist_ok=True)
json.dump(results, open("results/judge_calibration.json", "w"), indent=2)
