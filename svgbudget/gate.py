"""Layer 1, "the gate": is the model's answer a valid drawing that follows the rules?

check(text, max_shapes) runs the rules in order and stops at the first one that fails, so every
drawing gets exactly one reason: ok, no_svg, parse_error, tag_<name>, over_budget, blank, ...
"""
import re
import xml.etree.ElementTree as ET

import numpy as np

from svgbudget.render import render_svg

SVG_RE = re.compile(r"<svg\b.*?</svg>", re.DOTALL | re.IGNORECASE)
SVG_NS = "http://www.w3.org/2000/svg"
SHAPE_TAGS = {"circle", "ellipse", "rect", "line", "polyline", "polygon", "path"}
# Elements that draw nothing themselves. Everything else (text, use, image, style, pattern, ...) is rejected.
STRUCTURE_TAGS = {"svg", "g", "defs", "linearGradient", "radialGradient", "stop", "title", "desc"}
ALLOWED_TAGS = SHAPE_TAGS | STRUCTURE_TAGS
URL_RE = re.compile(r"url\(\s*([^)]*?)\s*\)", re.IGNORECASE)
MOVETO_RE = re.compile(r"[Mm]")
MIN_INK_FRACTION = 0.01          # under 1% non-white pixels counts as a blank canvas


def extract_svg(text):
    """The first <svg>...</svg> block in the model's answer, or None. Ignores reasoning before </think>."""
    if "</think>" in text:
        text = text.split("</think>")[-1]
    blocks = SVG_RE.findall(text)
    return blocks[0] if blocks else None


def local_name(tag):
    """'{http://www.w3.org/2000/svg}circle' -> 'circle'."""
    return tag.split("}", 1)[-1]


def parse_svg(svg):
    """Parse SVG text. Returns (root, None, "") on success or (None, reason, detail) on failure."""
    if re.search(r"<!(DOCTYPE|ENTITY)", svg, re.IGNORECASE):
        return None, "doctype", ""
    try:
        root = ET.fromstring(svg)
    except ET.ParseError as e:
        return None, "parse_error", str(e)
    if local_name(root.tag) != "svg":
        return None, "root_not_svg", root.tag
    return root, None, ""


def check_elements(root):
    """Only allowed elements and safe attributes. Returns (None, "") if fine, else (reason, detail)."""
    for el in root.iter():
        if el.tag.startswith("{") and not el.tag.startswith("{" + SVG_NS + "}"):
            return "foreign_namespace", el.tag
        name = local_name(el.tag)
        if name not in ALLOWED_TAGS:
            return f"tag_{name}", ""
        for attr, value in el.attrib.items():
            a = local_name(attr)
            if a == "href":
                return "href", ""
            if a.lower().startswith("on"):
                return "event_handler", a
            for target in URL_RE.findall(value):
                if not target.strip("'\"").startswith("#"):
                    return "external_url", target
    return None, ""


def count_shapes(root):
    """Shapes counted against the budget: each shape element is 1, and each <path> subpath (M/m) is 1."""
    n = 0
    for el in root.iter():
        name = local_name(el.tag)
        if name == "path":
            n += max(1, len(MOVETO_RE.findall(el.get("d", ""))))
        elif name in SHAPE_TAGS:
            n += 1
    return n


def ink_fraction(img):
    """Share of pixels that are visibly not white (faint anti-aliasing is ignored)."""
    a = np.asarray(img, dtype=np.int16)
    return float((np.abs(255 - a).sum(axis=2) > 30).mean())


def check(text, max_shapes):
    """Run every rule. Returns {ok, reason, detail, n_shapes, image}; image is only set when ok."""
    out = {"ok": False, "reason": None, "detail": "", "n_shapes": None, "image": None}

    svg = extract_svg(text)
    if svg is None:
        return {**out, "reason": "no_svg"}

    root, reason, detail = parse_svg(svg)
    if reason:
        return {**out, "reason": reason, "detail": detail}

    reason, detail = check_elements(root)
    if reason:
        return {**out, "reason": reason, "detail": detail}

    n = count_shapes(root)
    out["n_shapes"] = n
    if n == 0:
        return {**out, "reason": "no_shapes"}
    if n > max_shapes:
        return {**out, "reason": "over_budget", "detail": f"{n}>{max_shapes}"}

    try:
        img = render_svg(svg)
    except Exception as e:
        return {**out, "reason": "render_error", "detail": f"{type(e).__name__}: {e}"[:200]}
    ink = ink_fraction(img)
    if ink < MIN_INK_FRACTION:
        return {**out, "reason": "blank", "detail": f"ink={ink:.4f}"}

    return {**out, "ok": True, "reason": "ok", "image": img}
