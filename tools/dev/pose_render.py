"""Renders the built outfit in an animation pose (runs inside Blender):
  blender -b --python tools/dev/pose_render.py -- <work> <out_data> <png_prefix> <anim> <frame_frac> [preset]
      [projects, default Wrap,Loincloth,Sandals]"""
import os
import pickle
import sys

TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("lib", "pieces", ""):
    sys.path.insert(0, os.path.join(TOOLS, sub))
import bpy  # noqa: E402
import mk_testpresets  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1:]
WORK, DATA, OUT, ANIM, FRAC = argv[0], argv[1], argv[2], argv[3], float(argv[4])
PRESET = argv[5] if len(argv) > 5 else mk_testpresets.MAIN_PRESET
STEMS = argv[6].split(",") if len(argv) > 6 else ["Wrap", "Loincloth", "Sandals"]
sys.argv = [sys.argv[0], "--", WORK, DATA, OUT]
import clip_test as ct  # noqa: E402
import mk_blend  # noqa: E402
import mk_fit  # noqa: E402
import mk_preset  # noqa: E402

COL = {"Wrap": (0.93, 0.91, 0.86, 1), "WrapGold": (0.85, 0.65, 0.2, 1), "Loincloth": (0.55, 0.75, 0.95, 1),
       "LoinclothGold": (0.85, 0.65, 0.2, 1), "Sandals": (0.3, 0.2, 0.12, 1), "SandalsGold": (0.85, 0.65, 0.2, 1),
       "Briefs": (0.95, 0.95, 0.95, 1), "BriefsGold": (0.85, 0.65, 0.2, 1), "skin": (0.78, 0.6, 0.5, 1)}
presets = pickle.load(open(os.path.join(WORK, "ref", "presets.pkl"), "rb"))
body = mk_fit.Body(pickle.load(open(os.path.join(WORK, "ref", "body.pkl"), "rb")))
feet = mk_fit.Body(pickle.load(open(os.path.join(WORK, "ref", "feet.pkl"), "rb")), shape_names=("Feet",))
attrs = list(body.slider_attrs) + [a for a in feet.slider_attrs if a["name"] not in body.slider_names]
vals = mk_preset.slider_values(attrs, presets.get(PRESET, {}), 1.0)
poser = ct.Poser(os.path.join(WORK, "refs", "anims", "skeleton_female.hkx"))
pw = poser.pose(os.path.join(WORK, "refs", "anims", ANIM + ".hkx"), FRAC)
for stem in STEMS:
    for name, s in ct.load_project(stem).items():
        if name.endswith("Lining") or name.startswith("EV_"):
            continue
        v = poser.skin(s.morphed(vals), s.bones, s.w, pw)
        ob = mk_blend.mesh_object(stem + name, v, s.t)
        ob.color = COL["skin"] if name in ("BaseShape", "Feet") else COL.get(name, (0.6, 0.6, 0.6, 1))
mk_blend.setup_render(1000, color_type="OBJECT")
pel = pw[poser.sk.index["NPC Pelvis [Pelv]"]][0]
ZOOM = float(os.environ.get("POSE_ZOOM", "40"))
for view, d in (("front", (0, -1, 0)), ("left", (1, 0, 0)), ("q34", (0.6, -0.75, -0.25)), ("below", (0, -0.4, 0.9)),
                ("back", (0, 1, 0)), ("backq", (-0.5, 0.85, -0.15)), ("backup", (0, 0.75, -0.65))):
    cam = mk_blend.camera((pel[0], pel[1], pel[2] - 2), d, ZOOM)
    bpy.context.scene.render.filepath = "%s_%s.png" % (OUT, view)
    bpy.ops.render.render(write_still=True)
    bpy.data.objects.remove(cam)
