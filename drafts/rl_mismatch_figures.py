"""Build the mismatch article's explanatory SVGs; no empirical curves are generated."""

from html import escape
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "posts/rl-mismatch-mitigation/images"
OUT.mkdir(parents=True, exist_ok=True)

INK = "#172b40"
MUTED = "#526477"
BLUE = "#23659a"
ORANGE = "#ad591a"
PURPLE = "#74509b"
GREEN = "#28765a"


def text(x, y, value, size=18, color=INK, anchor="start", weight=400):
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
        f'text-anchor="{anchor}" font-weight="{weight}">{escape(value)}</text>'
    )


def start(width, height, title, desc):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f"<title id=\"title\">{escape(title)}</title><desc id=\"desc\">{escape(desc)}</desc>",
        '<style>text{font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}</style>',
        '<defs>',
        *[
            f'<marker id="{name}" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto">'
            f'<path d="M0 0L9 4.5L0 9Z" fill="{color}"/></marker>'
            for name, color in [("arrow", MUTED), ("blue", BLUE), ("purple", PURPLE)]
        ],
        "</defs>",
        f'<rect width="{width}" height="{height}" fill="#fff"/>',
    ]


def box(parts, x, y, w, h, title, lines, color=BLUE, fill="#eef5fa"):
    parts.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" '
        f'fill="{fill}" stroke="{color}" stroke-width="1.6"/>'
    )
    parts.append(text(x + w / 2, y + 32, title, 20, color, "middle", 650))
    for i, line in enumerate(lines):
        parts.append(text(x + w / 2, y + 61 + i * 24, line, 17, anchor="middle"))


def arrow(parts, path, color=MUTED, marker="arrow", dashed=False):
    dash = ' stroke-dasharray="6 5"' if dashed else ""
    parts.append(
        f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.1"'
        f'{dash} marker-end="url(#{marker})"/>'
    )


def write(name, parts):
    (OUT / name).write_text("\n".join(parts + ["</svg>"]) + "\n")


p = start(
    1120, 370,
    "One token: numerical disagreement and weight movement",
    "Illustrative probabilities at one fixed history. Rollout at v gives 0.20, trainer at v "
    "gives 0.21, and trainer at k gives 0.252. Same-version ratio 1.05 times weight-movement "
    "ratio 1.20 equals the current-to-behavior ratio 1.26.",
)
p += [
    text(40, 38, "One token, one fixed history", 25, weight=650),
    text(40, 66, "Illustrative probabilities · temperature 1 · no sampling truncation", 17, MUTED),
]
box(p, 40, 100, 270, 105, "Rollout · version v", ["qᵥ(a | h) = 0.20"], ORANGE, "#fff6ed")
box(p, 425, 100, 270, 105, "Trainer · version v", ["pᵥ(a | h) = 0.21"])
box(p, 810, 100, 270, 105, "Trainer · version k", ["pₖ(a | h) = 0.252"], PURPLE, "#f7f2fb")
p += [
    '<path d="M175 217V244H560V217" fill="none" stroke="#ad591a" stroke-width="2"/>',
    '<path d="M560 250V277H945V217" fill="none" stroke="#74509b" stroke-width="2"/>',
    text(365, 234, "Same-version ratio = 1.05", 18, ORANGE, "middle", 550),
    text(760, 267, "Weight-movement ratio = 1.20", 18, PURPLE, "middle", 550),
    '<rect x="250" y="302" width="620" height="44" rx="10" fill="#f2f6f9"/>',
    text(560, 331, "Current / behavior = 1.05 × 1.20 = 1.26", 21, anchor="middle", weight=600),
]
write("one-token-two-gaps.svg", p)

p = start(
    1200, 720,
    "Where mismatch mitigation enters an RL update",
    "Rollout forward records tokens, histories, and behavior probabilities. Trainer forward replays "
    "the tokens; its scores and the stored behavior probabilities enter the loss. Backward computes "
    "DGrad for earlier layers and WGrad for trainable tensors. WGrad enters the optimizer, which "
    "publishes weights back to both forward paths. The dashed forward-validation region fixes k=v "
    "before the update, with identical history and sampling settings; forward probability equality "
    "is measured, not assumed. Ordinary stale-data updates can use k>v.",
)
p += [
    text(40, 39, "Where mitigation enters the update", 27, weight=650),
    text(40, 68, "Forward agreement, loss correction, backward accuracy, and update retention are separate.", 17, MUTED),
    '<rect x="28" y="211" width="722" height="167" rx="15" fill="#fbfcfe" stroke="#94a3b8" stroke-dasharray="6 5"/>',
    text(390, 231, "Before-update check: k = v", 17, MUTED, "middle", 550),
    text(390, 252, "Same history and sampling settings", 16, MUTED, "middle"),
]
box(p, 420, 95, 300, 88, "Weight views / versions", ["Trainer k · rollout v"], PURPLE, "#f7f2fb")
box(p, 840, 95, 320, 117, "Optimizer", ["Master weights and moments", "Retain and apply WGrad"], PURPLE, "#f7f2fb")
box(p, 50, 260, 260, 100, "Rollout FProp", ["Generate from behavior q"], ORANGE, "#fff6ed")
box(p, 430, 260, 290, 100, "Trainer FProp", ["Score the recorded tokens"])
box(p, 50, 425, 260, 124, "Recorded experience", ["History h · action a", "Behavior log q(a | h)"], ORANGE, "#fff6ed")
box(p, 430, 425, 290, 124, "Loss and correction", ["PPO + TIS / masks", "SC · sampler KL · direct IS"])
box(p, 840, 350, 320, 200, "Trainer backward", ["DGrad → earlier layers", "WGrad → trainable tensors", "Validate precision / derivative"], GREEN, "#eff8f3")
box(p, 430, 603, 290, 78, "Reward / advantage", [], MUTED, "#f4f6f8")
arrow(p, "M420 139H180V255", PURPLE, "purple")
arrow(p, "M575 183V255", PURPLE, "purple")
arrow(p, "M840 139H725", PURPLE, "purple")
arrow(p, "M180 360V420")
arrow(p, "M310 454H367V310H425", BLUE, "blue")
arrow(p, "M310 510H425")
arrow(p, "M575 360V420", BLUE, "blue")
arrow(p, "M720 487H835")
arrow(p, "M1000 350V217", PURPLE, "purple")
arrow(p, "M575 603V554")
p += [
    text(260, 123, "Export / install", 16, PURPLE),
    text(775, 124, "Next version", 15, PURPLE, "middle"),
    text(383, 394, "Replay", 15, BLUE),
    text(775, 470, "Loss signal", 15, MUTED, "middle"),
    text(1040, 283, "WGrad", 17, PURPLE),
    text(50, 609, "Forward alignment", 18, BLUE, weight=650),
    text(50, 637, "Quantizers · kernels · routes · state", 16, MUTED),
    text(840, 609, "Backward / optimizer precision", 18, GREEN, weight=650),
    text(840, 637, "No corresponding rollout backward", 16, MUTED),
    text(600, 706, "Validation fixes k = v; stale updates can use k > v. Forward agreement is measured, not assumed.", 15, MUTED, "middle"),
]
write("update-interventions.svg", p)

p = start(
    1120, 380,
    "Aligned forward values and a separately validated backward",
    "At identical weights, history, and sampling settings, compare rollout and trainer log-probabilities. "
    "Their forward operations use aligned inference arithmetic. The trainer loss feeds a separately "
    "validated backward, which supplies gradients to the optimizer. Wider master weights and moments "
    "can coexist with the aligned forward. Rollout has no backward counterpart.",
)
p += [
    text(40, 39, "One forward reference; a separately validated backward", 26, weight=650),
    text(40, 72, "Agreement check: same weights, history, and sampling settings", 18, MUTED),
]
box(p, 40, 120, 300, 105, "Rollout forward", ["Aligned inference arithmetic", "Recorded log qᵥ"], ORANGE, "#fff6ed")
box(p, 430, 120, 300, 105, "Trainer forward", ["Aligned inference arithmetic", "Recomputed log pᵥ"])
box(p, 820, 120, 260, 105, "Trainer backward", ["DGrad / WGrad", "Validated separately"], GREEN, "#eff8f3")
box(p, 820, 270, 260, 85, "Optimizer", ["Master weights / moments"], PURPLE, "#f7f2fb")
arrow(p, "M340 173H425", dashed=True)
arrow(p, "M730 173H815")
arrow(p, "M950 225V265", PURPLE, "purple")
p += [
    text(385, 153, "Compare", 15, MUTED, "middle"),
    text(775, 153, "Via loss", 15, MUTED, "middle"),
    text(40, 282, "Forward equality does not establish gradient correctness.", 20, BLUE, weight=600),
    text(40, 316, "Backward can use wider arithmetic and a quantization surrogate.", 18, MUTED),
    text(40, 348, "Rollout has no corresponding backward or optimizer.", 18, MUTED),
]
write("tii-forward-backward.svg", p)

p = start(
    1200, 600,
    "A response can become stale before it enters a queue",
    "Illustrative timeline. Learner versions advance from 40 to 43 during response generation, a "
    "tool wait, reward processing, and queue residence. An optional mid-response refresh changes "
    "the generating version from 40 to 41. Tokens retain generating probabilities and versions. "
    "The time axis is schematic, not measured.",
)
p += [
    text(40, 40, "Staleness begins before the queue", 27, weight=650),
    text(40, 70, "Illustrative schedule · widths are not measured durations", 17, MUTED),
    text(40, 130, "Learner", 19, weight=650),
    text(40, 245, "Response", 19, weight=650),
    text(40, 345, "Metadata", 19, weight=650),
    '<path d="M180 140H1150" stroke="#94a3b8" stroke-width="2" marker-end="url(#arrow)"/>',
]
for x, label in [(180, "v40"), (440, "v41"), (735, "v42"), (1010, "v43")]:
    p += [
        f'<circle cx="{x}" cy="140" r="5" fill="{PURPLE}"/>',
        text(x, 120, label, 18, PURPLE, "middle", 600),
        f'<path d="M{x} 151V298" stroke="#e0d8e9" stroke-dasharray="4 5"/>',
    ]
segments = [
    (180, 170, "Generate", "version 40", "#fff1e4", ORANGE),
    (350, 190, "Tool wait", "no policy action", "#f2f5f8", MUTED),
    (540, 220, "Generate", "version 41", "#eaf3fa", BLUE),
    (760, 170, "Reward", "processing", "#f2f5f8", MUTED),
    (930, 150, "Queue", "completed data", "#f2f5f8", MUTED),
]
for x, w, title, subtitle, fill, color in segments:
    p += [
        f'<rect x="{x}" y="209" width="{w}" height="78" rx="7" fill="{fill}" stroke="{color}"/>',
        text(x + w/2, 238, title, 18, color, "middle", 600),
        text(x + w/2, 265, subtitle, 15, MUTED, "middle"),
    ]
arrow(p, "M540 174V204", PURPLE, "purple")
p += [text(555, 184, "Optional refresh", 15, PURPLE)]
arrow(p, "M1080 248H1140")
p += [
    text(1120, 319, "Train", 18, PURPLE, "middle", 600),
    '<rect x="180" y="322" width="900" height="70" rx="10" fill="#f8fafc" stroke="#cbd5e1"/>',
    text(205, 350, "Early tokens: recorded q₄₀(a | h) and v40", 17, ORANGE),
    text(205, 376, "Later tokens: recorded q₄₁(a | h) and v41", 17, BLUE),
    '<path d="M180 419V434H930V419" fill="none" stroke="#526477" stroke-width="2"/>',
    '<path d="M930 419V434H1080V419" fill="none" stroke="#526477" stroke-width="2"/>',
    text(555, 465, "Generation and waiting before queue entry", 18, MUTED, "middle", 550),
    text(1005, 465, "In queue", 18, MUTED, "middle", 550),
    text(40, 523, "A refresh is one scheduling choice. Other systems pin one version for the full response.", 17, MUTED),
    text(40, 551, "Version age, wall-clock age, and policy distance measure different things.", 17, MUTED),
    text(40, 579, "Retaining old cached state creates hybrid behavior; record its actual sampling probabilities.", 17, MUTED),
]
write("stale-response-timeline.svg", p)

print(f"Generated four explanatory figures in {OUT}")
