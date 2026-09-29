#!/usr/bin/env python
"""Schematic of one MD step and the outer supervision loop, with the modules changed in this work in green and the
measured gains as badges.  Emits two SVG variants from one geometry:
  --widget  : classes of the inline-widget design system (c-green / c-gray / t / ts / th), printed to stdout
  (default) : explicit colours + Helvetica, written to ../fig7_pipeline.svg and converted to ../fig7_pipeline.pdf (svglib)
Numbers (jobs in PROGRESS.md): LAMMPS overhead 0.25 -> 0.002 s/step and V1 step 0.434 -> 0.283 s (job72); resident memory
1.55 MB -> 4.57 kB/atom (job01, limit scans); N_max 86,016 -> 2.86e7 (job01, limit scans); exact kernels 0.95e5 -> 2.35e5
(fast layers + tabulated radial + graphs); student 2.25e5 -> 3.97e6 = NEP89/4.9 (job83); production 4.27e6 (job84 cycle 1).
"""
import os
import sys

WIDGET = "--widget" in sys.argv
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

G = dict(fill="#EAF3DE", stroke="#3B6D11", title="#27500A", sub="#3B6D11")      # green ramp 50 / 600 / 800 / 600
N = dict(fill="#F1EFE8", stroke="#5F5E5A", title="#444441", sub="#5F5E5A")      # gray ramp
PILL = dict(fill="#C0DD97", text="#173404")                                     # green 100 / 900
FONT = "" if WIDGET else ' font-family="Helvetica, Arial, sans-serif"'


def txt(x, y, s, cls, color, weight=400, size=14, anchor="middle"):
    if WIDGET:
        return f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{s}</text>'
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" font-weight="{weight}" fill="{color}"'
            f'{FONT}>{s}</text>')


def box(x, y, w, h, title, sub, green):
    c = G if green else N
    cls = "c-green" if green else "c-gray"
    if WIDGET:
        r = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4"/>'
        body = r + txt(x + w / 2, y + 23, title, "th", c["title"]) + txt(x + w / 2, y + 41, sub, "ts", c["sub"])
        return f'<g class="{cls}">{body}</g>'
    r = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" fill="{c["fill"]}" stroke="{c["stroke"]}" stroke-width="0.7"/>'
    return r + txt(x + w / 2, y + 23, title, "th", c["title"], 500, 14) + txt(x + w / 2, y + 41, sub, "ts", c["sub"], 400, 12)


def pill(cx, y, s):
    w = int(len(s) * 6.6) + 18
    x = cx - w / 2
    if WIDGET:
        return (f'<g class="c-green"><rect x="{x:.0f}" y="{y}" width="{w}" height="20" rx="10"/>'
                f'{txt(cx, y + 14, s, "ts", PILL["text"])}</g>')
    return (f'<rect x="{x:.0f}" y="{y}" width="{w}" height="20" rx="10" fill="{PILL["fill"]}"/>'
            + txt(cx, y + 14, s, "ts", PILL["text"], 500, 12))


def arrow(x1, y1, x2, y2):
    st = 'class="arr"' if WIDGET else 'stroke="#5F5E5A" stroke-width="1.2" fill="none"'
    return f'<path d="M{x1} {y1} L{x2} {y2}" {st} marker-end="url(#arrow)"/>'


def dashed(x, y, w, h, label):
    st = ('stroke="var(--b)"' if WIDGET else 'stroke="#888780"') + ' fill="none" stroke-width="0.7" stroke-dasharray="4 3"'
    lab = txt(x + 10, y + 16, label, "ts", "#444441", 500, 12, "start")
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" {st}/>' + lab


def build():
    e = []
    e.append('<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" '
             'orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" '
             'stroke-linecap="round" stroke-linejoin="round"/></marker></defs>')
    # ---- row 1: one MD step in LAMMPS (5 boxes, w 108, gap 15)
    y1, h = 80, 56
    xs = [40, 163, 286, 409, 532]
    row1 = [("Integrate", "half step", False), ("Neighbours", "list, ghosts", False), ("Pair data", "stays on GPU", True),
            ("Force call", "three passes", True), ("Integrate", "tally, step", False)]
    e.append(dashed(30, 66, 620, 84, "One MD step in LAMMPS (Kokkos, ML-IAP)"))
    for x, (t, s, g) in zip(xs, row1):
        e.append(box(x, y1 + 8, 108, h, t, s, g))
    for a, b in zip(xs, xs[1:]):
        e.append(arrow(a + 108, y1 + 8 + h / 2, b, y1 + 8 + h / 2))
    e.append(pill(340, 42, "LAMMPS overhead 0.25 to 0.002 s per step; step 0.43 to 0.28 s"))
    # ---- row 2: the force call, three passes
    e.append(arrow(463, y1 + 8 + h, 463, 186))
    e.append(dashed(30, 186, 620, 140, "Force call: exact three-pass evaluation of MACE (this work)"))
    y2 = 250
    xs2 = [56, 172, 288, 404, 520]
    row2 = [("Pass A", "layer 1 fwd", True), ("Exchange", "ghost H1", True), ("Pass B", "layer 2 bwd", True),
            ("Exchange", "ghost G1", True), ("Pass C", "layer 1 bwd", True)]
    for x, (t, s, g) in zip(xs2, row2):
        e.append(box(x, y2, 104, h, t, s, g))
    for a, b in zip(xs2, xs2[1:]):
        e.append(arrow(a + 104, y2 + h / 2, b, y2 + h / 2))
    e.append(pill(150, 210, "4.6 kB per atom (×340 less)"))
    e.append(pill(340, 210, "28.6 M atoms on one GPU"))
    e.append(pill(533, 210, "kernels ×2.5, graphs, compile"))
    # ---- row 3: outer supervision loop
    e.append(dashed(30, 356, 620, 150, "Outer loop: on-the-fly supervision by the exact model (this work)"))
    y3 = 420
    xs3 = [50, 200, 350, 500]
    row3 = [("Student MD", "one segment", True), ("Teacher check", "exact rerun", True),
            ("Retrain", "carve clusters", True), ("Restart", "next segment", True)]
    for x, (t, s, g) in zip(xs3, row3):
        e.append(box(x, y3, 130, h, t, s, g))
    for a, b in zip(xs3, xs3[1:]):
        e.append(arrow(a + 130, y3 + h / 2, b, y3 + h / 2))
    st = 'class="arr"' if WIDGET else 'stroke="#5F5E5A" stroke-width="1.2" fill="none"'
    e.append(f'<path d="M565 {y3 + h} L565 {y3 + h + 14} L115 {y3 + h + 14} L115 {y3 + h + 2}" {st} '
             'marker-end="url(#arrow)"/>')
    e.append(pill(190, 380, "student ×18 vs exact model (NEP89 / 4.9)"))
    e.append(pill(470, 380, "checks cost about 1.5% of the run"))
    # ---- legend + overall
    yl = 530
    for x, c, lab in ((40, G, "changed or new in this work"), (260, N, "unchanged LAMMPS and MACE")):
        if WIDGET:
            e.append(f'<g class="{"c-green" if c is G else "c-gray"}"><rect x="{x}" y="{yl - 11}" width="14" height="14" rx="3"/></g>')
        else:
            e.append(f'<rect x="{x}" y="{yl - 11}" width="14" height="14" rx="3" fill="{c["fill"]}" stroke="{c["stroke"]}" stroke-width="0.7"/>')
        e.append(txt(x + 20, yl, lab, "ts", "#444441", 400, 12, "start"))
    e.append(pill(340, yl + 20, "Overall: 4.3 M atom-steps/s at 1 M atoms (stock model: out of memory above 86 k)"))
    H = yl + 20 + 20 + 30
    if WIDGET:
        head = (f'<svg width="100%" viewBox="0 0 680 {H}" role="img"><title>MD pipeline with the modules changed in this '
                'work</title><desc>One LAMMPS MD step, the three-pass MACE force call and the outer supervision loop; '
                'green modules were changed, badges give the measured gains.</desc>')
    else:
        head = f'<svg xmlns="http://www.w3.org/2000/svg" width="680" height="{H}" viewBox="0 0 680 {H}">'
    return head + "".join(e) + "</svg>"


svg = build()
if WIDGET:
    sys.stdout.reconfigure(encoding="utf-8")
    print(svg)
else:
    p = os.path.join(OUT, "fig7_pipeline.svg")
    open(p, "w", encoding="utf-8", newline="\n").write(svg)
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPDF
    renderPDF.drawToFile(svg2rlg(p), os.path.join(OUT, "fig7_pipeline.pdf"))
    print("wrote fig7_pipeline.svg/.pdf")
