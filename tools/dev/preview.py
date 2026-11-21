"""Development preview: builds pieces and renders them on the UBE body (runs inside Blender).
  blender -b --factory-startup --python tools/dev/preview.py -- <work> <out_prefix> <pieces,comma> [preset|none]
          [weight] [views,comma] [center_z] [scale]"""
import importlib
import os
import pickle
import sys
import time

import numpy as np

TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("lib", "pieces", ""):
    sys.path.insert(0, os.path.join(TOOLS, sub))
import bpy  # noqa: E402

import mk_blend  # noqa: E402
import mk_fit  # noqa: E402
import mk_preset  # noqa: E402

COLORS = {"linen": (0.93, 0.91, 0.86, 1), "lining": (0.8, 0.76, 0.7, 1), "edge": (0.25, 0.17, 0.12, 1),
          "leather": (0.22, 0.15, 0.11, 1), "gold": (0.85, 0.65, 0.2, 1), "brass": (0.85, 0.65, 0.2, 1),
          "gem": (0.75, 0.15, 0.05, 1), "stone": (0.08, 0.08, 0.1, 1), "cord": (0.5, 0.3, 0.15, 1),
          "skin": (0.75, 0.62, 0.55, 1)}
argv = sys.argv[sys.argv.index("--") + 1:]
work, out, names = argv[0], argv[1], argv[2].split(",")
preset = argv[3] if len(argv) > 3 else "none"
weight = float(argv[4]) if len(argv) > 4 else 1.0
views = argv[5].split(",") if len(argv) > 5 else ["front", "left", "back", "q34"]
cz = float(argv[6]) if len(argv) > 6 else 70.0
scale = float(argv[7]) if len(argv) > 7 else 120.0
t0 = time.time()


def log(m):
    print("[%6.1fs] %s" % (time.time() - t0, m), flush=True)


ref = pickle.load(open(os.path.join(work, "ref", "body.pkl"), "rb"))
body = mk_fit.Body(ref)
ppath = os.path.join(work, "proxy.pkl")
import piece_wrap  # noqa: E402
if os.path.exists(ppath):
    proxy = pickle.load(open(ppath, "rb"))
    proxy.tree = mk_blend.bvh(proxy.v, proxy.t)
else:
    proxy = piece_wrap.make_proxy(body)
    tree, proxy.tree = proxy.tree, None
    pickle.dump(proxy, open(ppath, "wb"))
    proxy.tree = tree
log("body + proxy")
presets = pickle.load(open(os.path.join(work, "ref", "presets.pkl"), "rb"))
vals = mk_preset.slider_values(body.slider_attrs, presets.get(preset, {}), weight) if preset != "none" else {}
built = []
keep_tris = None
for n in names:
    mod = importlib.import_module("piece_" + n)
    import inspect
    kw = {"conform": False} if "conform" in inspect.signature(mod.build).parameters else {}
    res = mod.build(body, proxy, log, **kw)
    built += res["parts"]
    if res.get("skin"):
        sd = res["skin"][0]
        ob = mk_blend.mesh_object(sd["name"], sd["v"], sd["t"])
        ob.color = COLORS["skin"]
    if res.get("keep_tris") is not None:
        keep_tris = res["keep_tris"]
bv = body.apply_preset(vals)
bt = body.t if keep_tris is None else body.inv[body.orig_t[body.shape_names[0]]][keep_tris]
if "nobody" not in argv:
    skin = mk_blend.mesh_object("body", bv, bt)
    skin.color = COLORS["skin"]
    for key, shape in (("feet", "Feet"), ("hands", "Hands")):
        if key == "feet" and "sandals" in names:
            continue
        r2 = pickle.load(open(os.path.join(work, "ref", key + ".pkl"), "rb"))
        b2 = mk_fit.Body(r2, shape_names=(shape,))
        v2 = b2.apply_preset(mk_preset.slider_values(b2.slider_attrs, presets.get(preset, {}), weight)
                             if preset != "none" else {})
        ob = mk_blend.mesh_object(key, v2, b2.t)
        ob.color = COLORS["skin"]
    hd = pickle.load(open(os.path.join(work, "ref", "head.pkl"), "rb"))["shape"]
    ob = mk_blend.mesh_object("head", hd["v"], hd["t"])
    ob.color = COLORS["skin"]
for p in built:
    v = p.v.copy()
    if p.D is not None and vals:
        import mk_export
        for j, s in enumerate(p.sliders):
            if any(k in s for k in mk_export.NO_GARMENT):
                continue
            if s in vals and vals[s]:
                v += p.D[:, j] * vals[s]
    ob = mk_blend.mesh_object(p.name, v, p.t)
    ob.color = COLORS.get(p.mat, (0.5, 0.5, 0.9, 1))
log("objects")
mk_blend.setup_render(1000, color_type="OBJECT")
for name in views:
    cam = mk_blend.camera((0, 0, cz), mk_blend.VIEWS[name], scale)
    bpy.context.scene.render.filepath = "%s_%s.png" % (out, name)
    bpy.ops.render.render(write_still=True)
    bpy.data.objects.remove(cam)
log("rendered")
