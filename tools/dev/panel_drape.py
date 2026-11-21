"""Drape of the Loincloth's panels in animation poses (runs inside Blender):
  blender -b --factory-startup --python tools/dev/panel_drape.py -- <work> <out_data> <report_dir> [--tag name]
      [--render prefix]

FSMP holds the panels' chains close to their rest pose against the pelvis (springs of 14 against a pull of gravity of
about 1 on the whole chain), so in a still pose the hanging cloth stays with the pelvis while the part lying on the
body moves with the skin under it. Where that skin follows the thighs, the hips turning against the legs (the
player's hip-shot idles) dragged the panel's upper part sideways and the cloth bent into an S where the chains took
over (round 6). This test poses each panel of the built BodySlide project with the chains held to the pelvis and
measures, down its two edges and its middle, the cloth's sideways displacement relative to the pelvis: the sharpest
bend (second difference over 1.6 units of height) and the largest shift between any height and the hem.
Poses: the player's Gulan0 idles, walk and run, and the base game's walk, run, sprint, sneak and sitting.
--render: textured views from behind and below (the in-game screenshot's angle) in the player's idles."""
import glob
import json
import os
import pickle
import sys

import numpy as np

TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("lib", "pieces", ""):
    sys.path.insert(0, os.path.join(TOOLS, sub))
argv = sys.argv[sys.argv.index("--") + 1:]
OPTS = {}
for key in ("--tag", "--render"):
    if key in argv:
        i = argv.index(key)
        OPTS[key] = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
WORK, DATA, OUT = argv[0], argv[1], argv[2]
sys.argv = [sys.argv[0], "--", WORK, DATA, OUT]
import clip_test as ct  # noqa: E402
import mk_blend  # noqa: E402
import mk_fit  # noqa: E402
import mk_geom  # noqa: E402
import mk_nif  # noqa: E402
import mk_preset  # noqa: E402
import mk_testpresets  # noqa: E402

MODS = os.environ.get("MK_MODS", r"C:\Modding\TrueNorth\mods")
OAR = r"meshes\actors\character\animations\OpenAnimationReplacer"
BIN = 0.4       # height of a row of the measure
WIN = 4         # rows of the bend's second difference (1.6 units)


def animations():
    base = os.path.join(WORK, "refs", "anims")
    out = [(os.path.join(base, a + ".hkx"), a, [(k + 0.5) / n for k in range(n)])
           for a, n in (("mt_walkforward", 6), ("mt_runforward", 4), ("mt_sprintforward", 3), ("sneak_idle", 2),
                        ("chair_idle", 3))]
    idles = sorted(glob.glob(os.path.join(MODS, "[[]NoDelete[]] Gulan0 - Idles (OAR, static poses)", OAR, "**",
                                          "*.hkx"), recursive=True))
    out += [(p, "gulan0 " + os.path.basename(p)[:-4], (0.5,)) for p in idles]
    walks = sorted(glob.glob(os.path.join(MODS, "[[]NoDelete[]] Gulan0 - Movement (OAR - base and alternatives)",
                                          OAR, "**", "mt_walkforward.hkx"), recursive=True))
    if walks:
        out.append((walks[0], "gulan0 walk", (0.125, 0.375, 0.625, 0.875)))
    run = os.path.join(MODS, "[NoDelete] Gulan0 - Female Movement (CATA)", OAR, "CATA", "Light Armor", "female",
                       "mt_runforward.hkx")
    if os.path.exists(run):
        out.append((run, "gulan0 run (CATA)", (0.125, 0.375, 0.625, 0.875)))
    return [a for a in out if os.path.exists(a[0])]


def columns(v, t, side):
    """Vertices of a panel's outer face (the Loincloth shape, below the belt) picked down its two edges and its
    middle: per row of height BIN the leftmost, the rightmost and the one nearest x = 0."""
    n = mk_geom.vertex_normals(v, t)
    z_top = 71.8 if side < 0 else 70.0
    sel = (v[:, 1] * side > 0.5) & (n[:, 1] * side > 0.5) & (v[:, 2] < z_top) & (v[:, 2] > 44) & \
        (np.abs(v[:, 0]) < 7.5)
    idx = np.nonzero(sel)[0]
    rows = np.floor((z_top - v[idx, 2]) / BIN).astype(int)
    cols = {"left": [], "mid": [], "right": []}
    for r in range(rows.max() + 1):
        k = idx[rows == r]
        if len(k) < 3:
            continue
        cols["left"].append(k[np.argmin(v[k, 0])])
        cols["right"].append(k[np.argmax(v[k, 0])])
        cols["mid"].append(k[np.argmin(np.abs(v[k, 0]))])
    return {c: np.array(i) for c, i in cols.items()}


def pelvis_T(poser, pw):
    bi = poser.sk.index["NPC Pelvis [Pelv]"]
    pp, pq = pw[bi]
    rp, rq = poser.ref[bi]
    R = ct.qmat(pq) @ ct.qmat(rq).T
    return R, np.array(pp) - R @ np.array(rp)


def drape(shape, cols, vals, poser, pw):
    v = shape.morphed(vals)
    R, tv = pelvis_T(poser, pw)
    out = {}
    for name, idx in cols.items():
        P = poser.skin(v[idx], shape.bones, shape.w[idx], pw)
        dx = ((P - tv) @ R - v[idx])[:, 0]
        bend = np.abs(dx[2 * WIN:] - 2 * dx[WIN:-WIN] + dx[:-2 * WIN])
        out[name] = (float(bend.max()), float(np.abs(dx - dx[-1]).max()))
    return max(b for b, _ in out.values()), max(s for _, s in out.values()), out


def main():
    os.makedirs(OUT, exist_ok=True)
    body = mk_fit.Body(pickle.load(open(os.path.join(WORK, "ref", "body.pkl"), "rb")))
    presets = pickle.load(open(os.path.join(WORK, "ref", "presets.pkl"), "rb"))
    configs = [(p, mk_preset.slider_values(list(body.slider_attrs), presets.get(p, {}), 1.0))
               for p in mk_testpresets.for_tool("panel_drape") if p in presets]
    configs += list(mk_preset.extra_samples(WORK))[:1]
    shape = ct.load_project("Loincloth")["Loincloth"]
    panels = {side: columns(shape.v, shape.t, s) for side, s in (("back", -1), ("front", 1))}
    poser = ct.Poser(os.path.join(WORK, "refs", "anims", "skeleton_female.hkx"))
    results = {"poses": {}, "worst": {}}
    for path, tag, fracs in animations():
        for frac in fracs:
            pw = poser.pose(path, frac)
            line = []
            for pname, vals in configs:
                for side, cols in panels.items():
                    b, s, per = drape(shape, cols, vals, poser, pw)
                    key = "%s %.2f | %s | %s" % (tag, frac, pname, side)
                    results["poses"][key] = {"bend": round(b, 3), "shift": round(s, 3),
                                             "columns": {c: [round(x, 3) for x in bs] for c, bs in per.items()}}
                    w = results["worst"].setdefault(side + " | " + tag, {"bend": 0.0, "shift": 0.0})
                    w["bend"], w["shift"] = round(max(w["bend"], b), 3), round(max(w["shift"], s), 3)
                    line.append("%s %s %.2f/%.2f" % (pname[:10], side, b, s))
            print("%-36s %.2f | %s" % (tag[:36], frac, "  ".join(line)), flush=True)
    for side in panels:
        bs = [r["bend"] for k, r in results["poses"].items() if k.endswith("| " + side)]
        ss = [r["shift"] for k, r in results["poses"].items() if k.endswith("| " + side)]
        idle = [r["bend"] for k, r in results["poses"].items() if k.endswith("| " + side) and k.startswith("gulan0 mt_")]
        results[side] = {"bend_max": round(max(bs), 3), "bend_mean": round(float(np.mean(bs)), 3),
                         "shift_max": round(max(ss), 3), "idle_bend_max": round(max(idle), 3) if idle else None}
        print("SUMMARY %s panel: sharpest bend max %.3f mean %.3f (player's idles %.3f), largest shift %.2f"
              % (side, max(bs), np.mean(bs), max(idle) if idle else -1, max(ss)), flush=True)
    name = "panel_drape%s.json" % ("_" + OPTS["--tag"] if "--tag" in OPTS else "")
    json.dump(results, open(os.path.join(OUT, name), "w"), indent=1)
    print("written", os.path.join(OUT, name), flush=True)
    if "--render" in OPTS:
        render(OPTS["--render"], body, configs[0][1], poser)


def render(prefix, body, vals, poser):
    """Textured views (the atlas's diffuse) of the body, the Loincloth (with the briefs) and the Wrap's skin from
    behind and below in the player's idles."""
    import bpy
    sys.path.insert(0, os.path.join(TOOLS, "dev"))
    sd = os.path.join(DATA, "CalienteTools", "BodySlide", "ShapeData", "Sun Dancer")
    atlas = os.path.join(WORK, "tex", "SunDancer_d.png")
    meshes = []
    for stem in ("Wrap", "Loincloth"):
        nif = mk_nif.NifFile(os.path.join(sd, stem + ".nif"))
        shapes = ct.load_project(stem)
        for s in nif.shapes:
            sh = mk_nif.read_shape(s)
            if sh["name"].startswith("EV_") or sh["name"].endswith("Lining"):
                continue
            meshes.append((stem + sh["name"], shapes[sh["name"]], sh["uv"], sh["name"] == "BaseShape"))
    idles = [a for a in animations() if a[1].startswith("gulan0 mt_")]
    for path, tag, _ in idles:
        pw = poser.pose(path, 0.5)
        for ob in list(bpy.data.objects):
            bpy.data.objects.remove(ob)
        for m in list(bpy.data.materials):
            bpy.data.materials.remove(m)
        mat = bpy.data.materials.new("atlas")
        mat.use_nodes = True
        tex = mat.node_tree.nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(atlas, check_existing=True)
        mat.node_tree.links.new(tex.outputs["Color"], mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"])
        skin = bpy.data.materials.new("skin")
        skin.use_nodes = True
        skin.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.72, 0.52, 0.4, 1)
        for name, s, uv, is_skin in meshes:
            v = poser.skin(s.morphed(vals), s.bones, s.w, pw)
            ob = mk_blend.mesh_object(name, v, s.t, uv=None if is_skin else uv)
            ob.data.materials.append(skin if is_skin else mat)
        sc = bpy.context.scene
        sc.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in [
            e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items] else "BLENDER_EEVEE"
        sc.render.resolution_x = sc.render.resolution_y = 900
        world = bpy.data.worlds.new("w")
        world.use_nodes = True
        world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.42, 0.4, 0.38, 1)
        world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.9
        sc.world = world
        for rot, energy in (((0.9, 0.2, 0.6), 3.5), ((1.2, -0.3, -2.4), 1.6), ((2.2, 0.0, 3.1), 1.4)):
            ld = bpy.data.lights.new("sun", "SUN")
            ld.energy = energy
            lo = bpy.data.objects.new("sun", ld)
            lo.rotation_euler = rot
            sc.collection.objects.link(lo)
        pel = pw[poser.sk.index["NPC Pelvis [Pelv]"]][0]
        cam = mk_blend.camera((pel[0], pel[1], pel[2] - 7), (0, 1, 0.3), 34)
        sc.render.filepath = "%s_%s.png" % (prefix, tag.split()[-1])
        bpy.ops.render.render(write_still=True)
        bpy.data.objects.remove(cam)


main()
