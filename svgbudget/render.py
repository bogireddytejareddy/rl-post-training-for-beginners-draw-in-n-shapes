"""Turning SVG code (and raw Quick, Draw! strokes) into images, and tiling images into grids."""
import io
import textwrap

import cairosvg
from PIL import Image, ImageDraw, ImageFont


def render_svg(svg, size=256):
    """Render SVG code to a size x size RGB image on a white background."""
    png = cairosvg.svg2png(bytestring=svg.encode("utf-8"), output_width=size, output_height=size,
                           background_color="white", unsafe=False)
    return Image.open(io.BytesIO(png)).convert("RGB")


def sketch_to_svg(drawing, size=256, margin=12, stroke_width=3):
    """Quick, Draw! strokes -> SVG polylines, centred and scaled to fit (used to show human sketches to judges)."""
    xs = [x for stroke in drawing for x in stroke[0]]
    ys = [y for stroke in drawing for y in stroke[1]]
    w, h = max(xs) - min(xs), max(ys) - min(ys)
    scale = (size - 2 * margin) / max(w, h, 1)
    ox = (size - w * scale) / 2 - min(xs) * scale
    oy = (size - h * scale) / 2 - min(ys) * scale
    lines = []
    for sx, sy in drawing:
        pts = [(x * scale + ox, y * scale + oy) for x, y in zip(sx, sy)]
        if len(pts) == 1:
            pts = pts * 2                    # a dot needs two points to render
        p = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        lines.append(f'<polyline points="{p}" fill="none" stroke="black" stroke-width="{stroke_width}" '
                     f'stroke-linecap="round" stroke-linejoin="round"/>')
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}">' + "".join(lines) + "</svg>"


def render_grid(images, labels, cols=5, wrap=34, label_lines=2):
    """Tile same-size images into a grid with a short text label under each one."""
    w, h = images[0].size
    try:
        font = ImageFont.load_default(size=12)
    except TypeError:                         # older Pillow
        font = ImageFont.load_default()
    line_h = 15
    cell_h = h + label_lines * line_h + 8
    rows = (len(images) + cols - 1) // cols
    grid = Image.new("RGB", (cols * w, rows * cell_h), "white")
    draw = ImageDraw.Draw(grid)
    for i, (img, label) in enumerate(zip(images, labels)):
        x, y = (i % cols) * w, (i // cols) * cell_h
        grid.paste(img, (x, y))
        draw.rectangle([x, y, x + w - 1, y + h - 1], outline=(180, 180, 180))
        lines = [l for part in label.split("\n") for l in textwrap.wrap(part, wrap)][:label_lines]
        for j, line in enumerate(lines):
            draw.text((x + 4, y + h + 4 + j * line_h), line, fill="black", font=font)
    return grid
