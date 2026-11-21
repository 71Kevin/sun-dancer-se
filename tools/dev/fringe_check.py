"""Is the cup fringe visible? For every test preset (and the player's body), the Wrap's ShapeData is morphed and rays
are cast from each fringe vertex toward a viewer in front (level, and from a little above, as the game camera sees
her); a vertex is hidden when the cups, the band or the skin are in the way. Optional front renders. Runs in Blender:
  blender -b --factory-startup --python tools/dev/fringe_check.py -- <work> <shapedata_dir> [--render <png_prefix>]
      [--player <player_morphs.json>]
The fringe is told apart from the other gold of the Wrap by where it hangs (in front of the cups, above the band)."""
import os
import pickle
import sys

import numpy as np

TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("lib", "pieces", "", "dev"):
    sys.path.insert(0, os.path.join(TOOLS, sub))
argv = sys.argv[sys.argv.index("--") + 1:]
WORK, SD = argv[0], argv[1]
RENDER = argv[argv.index("--render") + 1] if "--render" in argv else None
PLAYER = argv[argv.index("--player") + 1] if "--player" in argv else None
sys.argv = [sys.argv[0], "--", WORK, SD, "."]
import clip_test as ct  # noqa: E402
import mk_blend  # noqa: E402
import mk_fit  # noqa: E402
import mk_geom  # noqa: E402
import mk_preset  # noqa: E402
import mk_testpresets  # noqa: E402

ct.SD = SD
VIEWS = {"level": np.array([0.0, 1.0, 0.0]), "above": mk_geom.normalize(np.array([0.0, 1.0, 0.35]))}


def fringe_mask(v):
    return (v[:, 2] > 90.8) & (v[:, 2] < 97.5) & (np.abs(v[:, 0]) > 1.2) & (np.abs(v[:, 0]) < 12.0) & (v[:, 1] > 2.0)


def main():
    presets = pickle.load(open(os.path.join(WORK, "ref", "presets.pkl"), "rb"))
    body = mk_fit.Body(pickle.load(open(os.path.join(WORK, "ref", "body.pkl"), "rb")))
    attrs = list(body.slider_attrs)
    configs = [(p, mk_preset.slider_values(attrs, presets.get(p, {}), 1.0)) for p in ct.PRESETS]
    if PLAYER:
        from diag_wrap_values import player_values
        configs.append(("player", player_values(PLAYER, 0.0)))
    w = ct.load_project("Wrap")
    gold = w["WrapGold"]
    sel = fringe_mask(gold.v)
    print("fringe vertices (reference shape): %d" % sel.sum())
    for name, vals in configs:
        occ_v = [w["BaseShape"].morphed(vals), w["Wrap"].morphed(vals)]
        occ_t = [w["BaseShape"].t, w["Wrap"].t]
        ov, ot = ct.merge(list(zip(occ_v, occ_t)))
        tree = mk_blend.bvh(ov, ot)
        gv = gold.morphed(vals)[sel]
        row = []
        for vname, d in VIEWS.items():
            dirs = np.repeat(d[None], len(gv), 0)
            _, _, _, dist = mk_blend.raycast(tree, gv + d * 0.02, dirs, 60.0)
            row.append("%s %3.0f%%" % (vname, 100.0 * np.mean(~np.isfinite(dist))))
        print("%-42s visible: %s" % (mk_testpresets.label(name), ", ".join(row)))
        if RENDER and name in ["player"] + mk_testpresets.for_tool("fringe_render", [mk_testpresets.ZEROED]):
            render(w, vals, "%s_%s" % (RENDER, name.split()[-1].strip("-").lower()))


def render(w, vals, path):
    import bpy
    mk_blend.clear_scene()
    for name, s in w.items():
        if name.endswith("Lining"):
            continue
        ob = mk_blend.mesh_object(name, s.morphed(vals), s.t)
        ob.color = (0.78, 0.6, 0.5, 1) if name == "BaseShape" else ((0.85, 0.65, 0.2, 1) if "Gold" in name
                                                                    else (0.92, 0.9, 0.84, 1))
    mk_blend.setup_render(900, color_type="OBJECT")
    sh = bpy.context.scene.display.shading
    sh.light = "STUDIO"
    sh.show_cavity = True
    for vname, (d, s) in {"front": ((0.0, -1.0, -0.3), 22), "side": ((-1.0, -0.2, -0.15), 20)}.items():
        cam = mk_blend.camera((0, 5.0, 92.0) if vname == "front" else (6, 5.0, 92.0), d, s)
        bpy.context.scene.render.filepath = "%s_%s.png" % (path, vname)
        bpy.ops.render.render(write_still=True)
        bpy.data.objects.remove(cam)


main()
