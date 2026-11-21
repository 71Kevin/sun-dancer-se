"""Which sliders make a garment shape lag behind the body (runs inside Blender).

  blender -b --factory-startup --python tools/dev/diag_sliders.py -- <work> <out_data> <player_morphs.json> <project>
      <shape> <zmin> <zmax>

For the garment vertices in the z band, the body point under each vertex (nearest point in the reference shape)
moves by the body's slider data; the garment vertex moves by its own. The per-slider difference along the body
normal, scaled by the player's slider value, shows which sliders open dents (negative: garment sinks in)."""
import json
import os
import pickle
import sys

import numpy as np

TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("lib", "pieces", "", "dev"):
    sys.path.insert(0, os.path.join(TOOLS, sub))
import mk_blend  # noqa: E402
import mk_fit  # noqa: E402
import clip_test  # noqa: E402
from diag_wrap_values import player_values  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1:]
WORK, DATA, MORPHS, PROJECT, SHAPE = argv[:5]
ZMIN, ZMAX = float(argv[5]), float(argv[6])


def main():
    body = mk_fit.Body(pickle.load(open(os.path.join(WORK, "ref", "body.pkl"), "rb")))
    clip_test.SD = os.path.join(DATA, "CalienteTools", "BodySlide", "ShapeData", "Sun Dancer")
    pr = clip_test.load_project(PROJECT)
    g = pr[SHAPE]
    sel = (g.v[:, 2] > ZMIN) & (g.v[:, 2] < ZMAX)
    gv = g.v[sel]
    tri, bc, loc, nrm, dist = body.project(gv)
    vals = player_values(MORPHS, 0.0)
    rows = []
    for k, s in enumerate(body.slider_names):
        x = vals.get(s, 0.0)
        bd = body.sample(tri, bc, body.D[:, k])
        gd = np.zeros_like(gv)
        if s in g.diffs:
            idx, d = g.diffs[s]
            full = np.zeros_like(g.v)
            full[idx] = d
            gd = full[sel]
        rel = ((gd - bd) * nrm).sum(1)                 # + garment moves out more than the skin, - sinks in
        if x:
            rows.append((s, x, float((rel * x).min()), float((rel * x).mean()), float(np.abs(bd).max()),
                         float(np.abs(gd).max())))
    rows.sort(key=lambda r: r[2])
    print("%d garment verts in z %.1f..%.1f" % (sel.sum(), ZMIN, ZMAX))
    print("%-34s %6s %8s %8s %8s %8s" % ("slider", "value", "min*x", "mean*x", "|body|", "|garm|"))
    for r in rows[:25]:
        print("%-34s %6.2f %8.3f %8.3f %8.3f %8.3f" % r)
    tot = np.zeros(len(gv))
    for k, s in enumerate(body.slider_names):
        x = vals.get(s, 0.0)
        if not x:
            continue
        bd = body.sample(tri, bc, body.D[:, k])
        gd = np.zeros_like(gv)
        if s in g.diffs:
            idx, d = g.diffs[s]
            full = np.zeros_like(g.v)
            full[idx] = d
            gd = full[sel]
        tot += ((gd - bd) * nrm).sum(1) * x
    print("combined: min %.3f mean %.3f (start offset mean %.3f)" % (tot.min(), tot.mean(), dist.mean()))


main()
