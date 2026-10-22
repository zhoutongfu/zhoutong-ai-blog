"""Generate the entropy post's original SVG figures without plotting dependencies.

Run from any directory: python3 drafts/entropy_blog_figures.py
These are analytical curves and a conceptual diagram, not empirical data.
"""

from pathlib import Path
from html import escape

OUT = Path(__file__).resolve().parents[1] / "posts/entropy-in-llm-rl/images"
OUT.mkdir(parents=True, exist_ok=True)


def text(x, y, value, size=16, color="#334155", anchor="start", weight="400"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
            f'text-anchor="{anchor}" font-weight="{weight}">{escape(value)}</text>')


def svg_start(width, height, title, description):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{escape(title)}</title><desc id="desc">{escape(description)}</desc>',
            '<style>text{font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}</style>',
            f'<rect width="{width}" height="{height}" fill="white"/>']


parts = svg_start(1080, 470, "Remove the contribution or cap its weight?",
    "Analytical positive-advantage gradient coefficients. PPO is r below 1.2 and zero above; "
    "Clip-Higher is r below 1.28 and zero above; CISPO is min(r,1.2) over the displayed range. "
    "At ratio 1.25 the coefficients are 0, 1.25, and 1.2. Threshold settings are illustrative.")
parts += [text(540, 34, "Remove the contribution—or cap its weight?", 23, "#0f172a", "middle", "650"),
          text(540, 61, "Coefficient on the token score gradient · positive advantage = +1", 16, anchor="middle")]
for i, (name, threshold, color, dash, detail) in enumerate([
    ("PPO", 1.2, "#2563a6", "", "Flat branch above 1.20"),
    ("Clip-Higher", 1.28, "#b95b12", "9 5", "Flat branch above 1.28"),
    ("CISPO", 1.2, "#7946a5", "10 4 2 4", "Weight capped at 1.20"),
]):
    left = 65 + i * 345
    width, top, bottom = 270, 160, 350
    px = lambda r: left + (r - .8) / .8 * width
    py = lambda c: bottom - c / 1.4 * (bottom - top)
    parts += [text(left + width/2, 105, name, 20, color, "middle", "650"),
              text(left + width/2, 130, detail, 15, anchor="middle")]
    for v in [0, .4, .8, 1.2]:
        parts.append(f'<path d="M{left} {py(v)}H{left+width}" stroke="#e2e8f0"/>')
        parts.append(text(left - 10, py(v) + 5, f"{v:g}", 13, anchor="end"))
    for r in [.8, 1, 1.2, 1.4, 1.6]:
        parts.append(f'<path d="M{px(r)} {bottom}v5" stroke="#94a3b8"/>')
        parts.append(text(px(r), bottom+25, f"{r:g}", 13, anchor="middle"))
    parts.append(f'<path d="M{left} {top}V{bottom}H{left+width}" fill="none" stroke="#94a3b8"/>')
    parts.append(f'<path d="M{px(1.25)} {top}V{bottom}" stroke="#94a3b8" stroke-dasharray="3 5"/>')
    style=f'fill="none" stroke="{color}" stroke-width="3.5" stroke-dasharray="{dash}"'
    if i < 2:
        parts.append(f'<path d="M{px(.8)} {py(.8)}L{px(threshold)} {py(threshold)}" {style}/>')
        parts.append(f'<path d="M{px(threshold)} {py(0)}H{px(1.6)}" {style}/>')
        for y in [py(threshold), py(0)]:
            parts.append(f'<circle cx="{px(threshold)}" cy="{y}" r="4.5" fill="white" stroke="{color}" stroke-width="2"/>')
    else:
        parts.append(f'<path d="M{px(.8)} {py(.8)}L{px(threshold)} {py(threshold)}H{px(1.6)}" {style}/>')
    c = [0, 1.25, 1.2][i]
    parts.append(f'<circle cx="{px(1.25)}" cy="{py(c)}" r="5" fill="{color}"/>')
    parts.append(text(px(1.25)+14, py(c)-13, f"{c:g}", 17, color, weight="650"))
    parts.append(text(left+width/2, 406, "Probability ratio r", 15, anchor="middle"))
parts += [text(540, 442, "Dotted guides: r = 1.25. Open circles mark undefined branch derivatives at clipping boundaries.", 14, anchor="middle"), '</svg>']
(OUT / "gradient-coefficients.svg").write_text("\n".join(parts)+"\n")

parts = svg_start(1080, 485, "Three places to change the RL feedback loop",
    "Current policy generates responses through sampling. Rewards and advantages drive the task update, "
    "which produces an updated policy together with auxiliary regularization. The updated policy generates "
    "the next batch. Sampling, task-update design, and regularization intervene at different points.")
parts += ['<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8Z" fill="#64748b"/></marker></defs>',
          text(540, 39, "Three places to change the RL feedback loop", 24, "#0f172a", "middle", "650")]


def box(x, y, width, height, title, lines, fill, stroke, color, number=None):
    parts.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="13" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    parts.append(text(x+width/2, y+32, title, 19, color, "middle", "650"))
    for j, line in enumerate(lines):
        parts.append(text(x+width/2, y+60+j*23, line, 16, anchor="middle"))
    if number:
        parts.append(f'<circle cx="{x+17}" cy="{y}" r="15" fill="{stroke}"/>')
        parts.append(text(x+17, y+5, number, 15, "white", "middle", "650"))


def arrow(d):
    parts.append(f'<path d="{d}" fill="none" stroke="#64748b" stroke-width="2.5" marker-end="url(#arrow)"/>')


box(30, 140, 160, 110, "Current policy", ["Token probabilities"], "#f8fafc", "#94a3b8", "#0f172a")
box(230, 125, 220, 140, "Rollout sampling", ["Which responses appear?", "Temperature · groups"], "#eef6fc", "#2563a6", "#174879", "1")
box(490, 140, 170, 110, "Reward signal", ["Rewards", "and advantages"], "#f8fafc", "#94a3b8", "#0f172a")
box(700, 125, 340, 140, "Policy-update design", ["Which gradients contribute?", "Weighting · clipping · selection"], "#fff6ed", "#b95b12", "#87400c", "2")
box(350, 325, 310, 110, "Regularization", ["What additional preference?", "Entropy bonus · anchor penalty"], "#f7f0fc", "#7946a5", "#603784", "3")
box(700, 325, 340, 110, "Updated policy", ["Task and auxiliary gradients", "jointly change parameters"], "#f8fafc", "#94a3b8", "#0f172a")
arrow("M190 195H225"); arrow("M450 195H485"); arrow("M660 195H695")
arrow("M870 265V320"); arrow("M660 380H695")
arrow("M700 410H690V465H110V255")
parts += [text(210, 446, "Next rollout batch", 15, anchor="middle"), '</svg>']
(OUT / "rl-feedback-loop.svg").write_text("\n".join(parts)+"\n")
print(f"Generated two original SVGs in {OUT}")
