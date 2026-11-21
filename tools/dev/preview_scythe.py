"""Development preview of the scythe (runs inside Blender): blender -b --python tools/dev/preview_scythe.py -- <png>"""
import os
import sys

TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("lib", "pieces"):
    sys.path.insert(0, os.path.join(TOOLS, sub))
import bpy  # noqa: E402

import mk_blend  # noqa: E402
import piece_scythe  # noqa: E402

COLORS = {"steel": (0.12, 0.12, 0.14, 1), "ember": (1.0, 0.55, 0.15, 1), "wood": (0.1, 0.07, 0.05, 1),
          "leather": (0.3, 0.18, 0.1, 1), "gold": (0.85, 0.65, 0.2, 1), "obsidian": (0.03, 0.03, 0.05, 1)}
out = sys.argv[sys.argv.index("--") + 1]
res = piece_scythe.build()
for p in res["parts"]:
    ob = mk_blend.mesh_object(p.name, p.v, p.t)
    ob.color = COLORS.get(p.mat, (0.5, 0.5, 0.9, 1))
mk_blend.setup_render(1000, color_type="OBJECT")
for name, d, c, s in (("side", (0, 0, -1), (-20, 5, 0), 120), ("head", (0, 0, -1), (-22, 40, 0), 60),
                      ("q34", (0.5, -0.3, -0.8), (-20, 5, 0), 120), ("edge", (-1, 0, 0), (-20, 5, 0), 120)):
    cam = mk_blend.camera(c, d, s)
    bpy.context.scene.render.filepath = "%s_%s.png" % (out, name)
    bpy.ops.render.render(write_still=True)
    bpy.data.objects.remove(cam)
