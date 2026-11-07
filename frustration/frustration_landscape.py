"""Maximum-frustration landscape figure (dependency-free SVG).

Panels share the y axis f in [0, 1/2], the maximum over signings of the least unsatisfied weight fraction:
  (a) K5-minor-free graphs of girth g: maximum 1/g (weighted; Proposition 5.2 of the manuscript).
  (b) K_{l,r}, r a multiple of 2^(l-1): (1/2)(1 - C(l-1, floor((l-1)/2)) / 2^(l-1)) (Graham-Sloane, sharp by Bowlin 2012).
  (c) K_n: floor((n-1)^2/4) / C(n,2) (Petersdorf 1966).
  (d) Z^d: lower bound max(1/4, 1/2 - 1/(2 sqrt d)) from the fully frustrated lattice; the maximum is known only at d = 2.
The SVG is square so that QuickLook renders it uncropped. Explanations live in the manuscript caption.
Usage: python frustration_landscape.py OUT.svg
"""
import math
import sys

SURFACE, INK, INK2, GRID, SERIES = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#2a78d6"
PW, PH = 400, 348  # plot area per panel
ML, MT, MR, MB = 80, 92, 26, 66  # panel margins (left, top: title + subtitle, right, bottom)
COLS, ROWS = 2, 2
W, H = COLS * (ML + PW + MR), ROWS * (MT + PH + MB)
SUB = '<tspan dy="5" font-size="15">{}</tspan><tspan dy="-5">{}</tspan>'
SUP = '<tspan dy="-8" font-size="15">{}</tspan><tspan dy="8">{}</tspan>'


def panel(ox, oy, title, subtitle, xs, ys, exact, xlabel, xticks, refs):
    out = []
    x0, x1 = min(xs[0], xticks[0]), max(xs[-1], xticks[-1])
    sx = lambda x: ox + ML + (x - x0) / (x1 - x0) * PW
    sy = lambda y: oy + MT + PH - y / 0.5 * PH
    out.append(f'<text x="{ox + 12}" y="{oy + 32}" font-size="21" font-weight="600" fill="{INK}">{title}</text>')
    out.append(f'<text x="{ox + 12}" y="{oy + 58}" font-size="17" fill="{INK2}">{subtitle}</text>')
    for yt in (0, 0.1, 0.2, 0.3, 0.4, 0.5):
        y = sy(yt)
        out.append(f'<line x1="{ox + ML}" x2="{ox + ML + PW}" y1="{y:.1f}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        out.append(f'<text x="{ox + ML - 10}" y="{y + 6:.1f}" font-size="18" fill="{INK2}" text-anchor="end">{yt:g}</text>')
    for xt in xticks:
        out.append(f'<text x="{sx(xt):.1f}" y="{oy + MT + PH + 24}" font-size="18" fill="{INK2}" text-anchor="middle">{xt}</text>')
    out.append(f'<text x="{ox + ML + PW / 2}" y="{oy + MT + PH + 52}" font-size="19" fill="{INK2}" text-anchor="middle">{xlabel}</text>')
    for yv, label in refs:
        y = sy(yv)
        out.append(f'<line x1="{ox + ML}" x2="{ox + ML + PW}" y1="{y:.1f}" y2="{y:.1f}" stroke="{INK2}" stroke-width="1.4" stroke-dasharray="6 5"/>')
        out.append(f'<text x="{ox + ML + PW - 4}" y="{y - 8:.1f}" font-size="17" fill="{INK2}" text-anchor="end">{label}</text>')
    pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, ys))
    out.append(f'<polyline points="{pts}" fill="none" stroke="{SERIES}" stroke-width="2.4" stroke-linejoin="round"/>')
    for x, y, ex in zip(xs, ys, exact):
        fill = SERIES if ex else SURFACE
        out.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="5.5" fill="{fill}" stroke="{SERIES}" stroke-width="2.4"/>')
    return out


def main(path):
    body = [f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>']
    g = list(range(3, 11))
    body += panel(0, 0, "(a) K" + SUB.format("5", "-minor-free, girth g"),
                  "weighted maximum 1/g",
                  g, [1 / x for x in g], [True] * len(g), "girth g", list(range(3, 11)), [])
    ls = list(range(2, 17))
    fb = [0.5 * (1 - math.comb(l - 1, (l - 1) // 2) / 2 ** (l - 1)) for l in ls]
    body += panel(ML + PW + MR, 0, "(b) K" + SUB.format("l,r", ", r a multiple of 2") + SUP.format("l−1", ""),
                  "Graham–Sloane bound, sharp (Bowlin 2012)",
                  ls, fb, [True] * len(ls), "smaller side l", [2, 4, 6, 8, 10, 12, 14, 16],
                  [(0.25, "loop cap 1/4"), (0.5, "1/2")])
    ns = list(range(3, 17))
    fk = [((n - 1) ** 2 // 4) / math.comb(n, 2) for n in ns]
    body += panel(0, MT + PH + MB, "(c) Complete graph K" + SUB.format("n", ""),
                  "Petersdorf 1966: all-negative K" + SUB.format("n", " is extremal"),
                  ns, fk, [True] * len(ns), "n", [3, 6, 9, 12, 16],
                  [(1 / 3, "loop cap 1/3"), (0.5, "1/2")])
    ds = list(range(2, 13))
    fz = [max(0.25, 0.5 - 0.5 / math.sqrt(d)) for d in ds]
    body += panel(ML + PW + MR, MT + PH + MB, "(d) Hypercubic lattice ℤ" + SUP.format("d", ""),
                  "lower bounds from the fully frustrated lattice",
                  ds, fz, [d == 2 for d in ds], "dimension d", [2, 4, 6, 8, 10, 12],
                  [(0.25, "loop cap 1/4"), (0.5, "1/2")])
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
           'font-family="Helvetica, Arial, sans-serif">' + "".join(body) + "</svg>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(svg)


if __name__ == "__main__":
    main(sys.argv[1])
