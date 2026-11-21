"""Quick look at the briefs while shaping them (runs inside Blender, builds only the briefs):
  blender -b --factory-startup --python tools/dev/briefs_preview.py -- <work> <png_prefix> [preset ...]
Renders the reference body (and each preset given, at weight 100) with the briefs in flat colours: the trim islands
bronze, the front and back cloth white, the rings gold. Views: front, back, left side, three-quarter, below."""
import os
import pickle
import sys

import numpy as np

TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("lib", "pieces", ""):
    sys.path.insert(0, os.path.join(TOOLS, sub))
import bpy  # noqa: E402

import mk_blend  # noqa: E402
import mk_fit  # noqa: E402
import mk_preset  # noqa: E402
import piece_briefs  # noqa: E402

argv = [a for a in sys.argv[sys.argv.index("--") + 1:] if not a.startswith("--")]
WORK, OUT = argv[0], argv[1]
PRESETS = argv[2:]


def colour(p):
    if p.mat in ("gold", "gem"):
        return (0.85, 0.62, 0.2, 1)
    if p.island.startswith("brieftrim"):
        return (0.45, 0.3, 0.16, 1)
    if p.island == "briefs_0":
        return (0.93, 0.91, 0.86, 1)
    if p.island == "briefs_1":
        return (0.86, 0.88, 0.93, 1)
    if p.island == "patch:edge":
        return (0.25, 0.16, 0.1, 1)
    return (0.7, 0.7, 0.68, 1)


def morph(v, D, names, vals):
    """The garment under a preset, without the sliders the export leaves out of garments (mk_export.NO_GARMENT)."""
    import mk_export
    x = np.array([0.0 if any(k in n for k in mk_export.NO_GARMENT) else vals.get(n, 0.0) for n in names])
    return v + np.einsum("s,nsd->nd", x, D) if np.any(x) else v


def main():
    body = mk_fit.Body(pickle.load(open(os.path.join(WORK, "ref", "body.pkl"), "rb")))
    proxy = pickle.load(open(os.path.join(WORK, "proxy.pkl"), "rb"))
    proxy.tree = mk_blend.bvh(proxy.v, proxy.t)
    parts, (v, t, labels) = piece_briefs.build_parts(body, proxy)
    pickle.dump([(p.name, p.island, p.mat, p.v, p.t, p.w, p.D) for p in parts],
                open(OUT + "_parts.pkl", "wb"))
    presets = pickle.load(open(os.path.join(WORK, "ref", "presets.pkl"), "rb"))
    configs = [("ref", {})] + [(p, mk_preset.slider_values(body.slider_attrs, presets.get(p, {}), 1.0))
                               for p in PRESETS if p in presets]
    for tag, vals in configs:
        for ob in list(bpy.data.objects):
            bpy.data.objects.remove(ob)
        bv = body.apply_preset(vals) if vals else body.v
        sk = mk_blend.mesh_object("body", bv, body.t)
        sk.color = (0.78, 0.62, 0.53, 1)
        for p in parts:
            ob = mk_blend.mesh_object(p.name, morph(p.v, p.D, proxy.slider_names, vals), p.t)
            ob.color = colour(p)
        mk_blend.setup_render(1000, color_type="OBJECT")
        safe = "".join(ch if ch.isalnum() else "_" for ch in tag)[:24]
        for view, d, c, s in (("front", (0, -1, 0), (0, 0, 68), 30), ("back", (0, 1, 0), (0, 0, 68), 30),
                              ("left", (1, 0, 0), (0, -2, 68), 30), ("q34", (0.6, -0.8, -0.05), (0, 0, 68), 30),
                              ("below", (0.0, -0.35, 0.94), (0, -1, 64), 22), ("backq", (-0.55, 0.83, 0.1), (0, 0, 68), 30)):
            cam = mk_blend.camera(c, d, s)
            bpy.context.scene.render.filepath = "%s_%s_%s.png" % (OUT, safe, view)
            bpy.ops.render.render(write_still=True)
            bpy.data.objects.remove(cam)
        if "--alone" in sys.argv:
            sk.hide_render = True
            for view, d, c, s in (("aloneunder", (0, 0, 1), (0, -2, 64), 16), ("alonebacklow", (0, 0.6, 0.8), (0, -4, 65), 16),
                                  ("aloneback", (0, 1, 0), (0, -4, 66), 16)):
                cam = mk_blend.camera(c, d, s)
                bpy.context.scene.render.filepath = "%s_%s_%s.png" % (OUT, safe, view)
                bpy.ops.render.render(write_still=True)
                bpy.data.objects.remove(cam)
            sk.hide_render = False


main()
