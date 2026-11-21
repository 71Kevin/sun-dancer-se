"""Close-up renders of the built outfit on the player's in-game body (runs inside Blender):

  blender -b --python tools/dev/render_player.py -- <work> <out_data> <png_prefix> <player_morphs.json>
      [anim frame_frac] [--projects Wrap,Loincloth,...] [--full-body]

The body is the BodySlide base preset plus the OBody morphs read from the save; with --full-body the untrimmed UBE
body is drawn instead of the Wrap's trimmed skin (shows where the garment sinks into the body)."""
import os
import pickle
import sys

TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("lib", "pieces", "", "dev"):
    sys.path.insert(0, os.path.join(TOOLS, sub))
import bpy  # noqa: E402
import numpy as np  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1:]
WORK, DATA, OUT, MORPHS = argv[:4]
rest = argv[4:]
ANIM = None
if rest and not rest[0].startswith("--"):
    ANIM, FRAC = rest[0], float(rest[1])
PROJECTS = ["Wrap", "Loincloth", "Bracelets", "Armlets", "Collar"]
if "--projects" in rest:
    PROJECTS = rest[rest.index("--projects") + 1].split(",")
FULL = "--full-body" in rest
NOSKIN = "--no-skin" in rest
sys.argv = [sys.argv[0], "--", WORK, DATA, OUT]
import clip_test as ct  # noqa: E402
import mk_blend  # noqa: E402
import mk_fit  # noqa: E402
from diag_wrap_values import player_values  # noqa: E402

ct.SD = os.path.join(DATA, "CalienteTools", "BodySlide", "ShapeData", "Sun Dancer")
GOLD = (0.85, 0.65, 0.2, 1)
vals = player_values(MORPHS, 0.0)
body = mk_fit.Body(pickle.load(open(os.path.join(WORK, "ref", "body.pkl"), "rb")))
poser = pw = None
if ANIM:
    poser = ct.Poser(os.path.join(WORK, "refs", "anims", "skeleton_female.hkx"))
    pw = poser.pose(os.path.join(WORK, "refs", "anims", ANIM + ".hkx"), FRAC)


def place(v, bones, w):
    return poser.skin(v, bones, w, pw) if poser else v


mk_blend.clear_scene()
if FULL:
    bv = place(body.apply_preset(vals), body.bones, body.w)
    ob = mk_blend.mesh_object("body", bv, body.t)
    ob.color = (0.78, 0.6, 0.5, 1)
for stem in PROJECTS:
    for name, s in ct.load_project(stem).items():
        if name.endswith("Lining") or name.startswith("EV_") or ((FULL or NOSKIN) and name == "BaseShape"):
            continue
        v = place(s.morphed(vals), s.bones, s.w)
        ob = mk_blend.mesh_object(stem + name, v, s.t)
        if name == "BaseShape":
            ob.color = (0.78, 0.6, 0.5, 1)
        elif "Gold" in name or "Chains" in name:
            ob.color = GOLD
        elif name.startswith("Loincloth"):
            ob.color = (0.9, 0.88, 0.82, 1)
        elif name == "Veils":
            ob.color = (0.95, 0.7, 0.3, 1)
        else:
            ob.color = (0.62, 0.8, 1.0, 1)
mk_blend.setup_render(1000, color_type="OBJECT")
sh = bpy.context.scene.display.shading
sh.light = "STUDIO"
sh.show_cavity = True
sh.cavity_type = "BOTH"
sh.show_specular_highlight = True
VIEWS = {"chest": ((0, 4.0, 92.0), (0.0, -1.0, -0.15), 22), "chest34": ((3, 4.0, 92.0), (-0.55, -0.8, -0.2), 22),
         "chestside": ((6, 3.0, 92.0), (-1.0, -0.15, -0.1), 22), "belly": ((0, 3.0, 72.0), (0.0, -1.0, -0.45), 18),
         "hips34": ((0, 0.0, 66.0), (-0.6, -0.8, -0.1), 34), "back": ((0, -3.0, 86.0), (0.0, 1.0, -0.1), 36), "full": ((0, 0, 60.0), (0.35, -1.0, -0.12), 70), "fullback": ((0, 0, 60.0), (-0.3, 1.0, -0.1), 70), "hipside": ((0, 0, 64.0), (-1.0, -0.2, -0.1), 34), "feet": ((0, 2, 10.0), (0.4, -1.0, -0.4), 22),
         "forearmL": ((-26.0, 2.0, 86.0), (0.2, -1.0, 0.1), 16),
         "wristL": ((-26.3, -1.0, 78.0), (0.75, -0.62, 0.15), 8), "wristLback": ((-26.3, -1.0, 78.0), (0.4, 0.9, 0.1), 8),
         "armL": ((-19.0, -3.0, 90.0), (1.0, 0.05, 0.0), 38), "armback": ((-14.0, -3.0, 88.0), (0.25, 1.0, 0.0), 40),
         "armfront": ((-16.0, 0.0, 90.0), (0.2, -1.0, 0.0), 38)}
ONLY_VIEWS = rest[rest.index("--views") + 1].split(",") if "--views" in rest else None
for view, (c, d, scale) in VIEWS.items():
    if ONLY_VIEWS and view not in ONLY_VIEWS:
        continue
    if poser and view.startswith("forearm"):
        continue
    cam = mk_blend.camera(c, d, scale)
    bpy.context.scene.render.filepath = "%s_%s.png" % (OUT, view)
    bpy.ops.render.render(write_still=True)
    bpy.data.objects.remove(cam)

