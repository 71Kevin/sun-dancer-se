"""Landmarks of the UBE reference body and quick renders (development helper, runs inside Blender).
  blender -b --factory-startup --python tools/probe_body.py -- <work> <out_png_prefix> [preset name] [weight]"""
import os
import pickle
import sys

import numpy as np

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(TOOLS, "lib"))
import mk_blend  # noqa: E402
import mk_fit  # noqa: E402
import mk_preset  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1:]
work, out = argv[0], argv[1]
preset = argv[2] if len(argv) > 2 else None
weight = float(argv[3]) if len(argv) > 3 else 1.0
ref = pickle.load(open(os.path.join(work, "ref", "body.pkl"), "rb"))
body = mk_fit.Body(ref)
presets = pickle.load(open(os.path.join(work, "ref", "presets.pkl"), "rb"))
vals = mk_preset.slider_values(body.slider_attrs, presets.get(preset, {}) if preset else {}, weight)
v = body.apply_preset(vals)
print("verts", len(v), "sliders", len(body.slider_names))
front = v[:, 1] > 0
for side in (-1, 1):
    m = front & (np.sign(v[:, 0]) == side) & (v[:, 2] > 85) & (v[:, 2] < 102)
    i = np.argmax(np.where(m, v[:, 1], -99))
    print("nipple", side, v[i].round(2))
m = front & (np.abs(v[:, 0]) < 1.5) & (v[:, 2] > 74) & (v[:, 2] < 86)
i = np.argmin(np.where(m, v[:, 1], 99))
print("navel", v[i].round(2))
for z in (110, 105, 100, 95, 90, 86, 82, 78, 74, 70, 66, 62, 55, 45, 35, 25, 15, 10):
    s = (np.abs(v[:, 2] - z) < 0.5)
    torso = s & (np.abs(v[:, 0]) < 16)
    p = v[torso]
    if len(p):
        print("z=%5.1f x[%6.2f %6.2f] y[%6.2f %6.2f]" % (z, p[:, 0].min(), p[:, 0].max(), p[:, 1].min(), p[:, 1].max()))
ob = mk_blend.mesh_object("body", v, body.t)
mk_blend.render_views(out, (0, 0, 70), 130, views=("front", "left", "back"), res=900)
