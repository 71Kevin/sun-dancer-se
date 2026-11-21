"""Textured preview renders of the built outfit on a preset (runs inside Blender):
  blender -b --factory-startup --python tools/dev/render_textured.py -- <work> <out_data> <png_prefix> [preset] [weight]
      [--views front,fringe,...]
Uses the BodySlide ShapeData (NIF + OSD) morphed with the preset (or, for a .json path, the body morphs written by
tools/dev/save_morphs.py), the painted atlas (cache\\tex\\*_d.png, its alpha on the sheer veils) and the scythe NIF.
The physics cloth is shown in its rest shape."""
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
import mk_nif  # noqa: E402
import mk_preset  # noqa: E402
import mk_testpresets  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1:]
VIEWS = None
if "--views" in argv:
    VIEWS = argv[argv.index("--views") + 1].split(",")
    argv = argv[:argv.index("--views")] + argv[argv.index("--views") + 2:]
# which of the three slot-52 items to show: Loincloth (with the briefs), LoinclothNoBriefs, Briefs or none
LOWER = "Loincloth"
if "--lower" in argv:
    LOWER = argv[argv.index("--lower") + 1]
    argv = argv[:argv.index("--lower")] + argv[argv.index("--lower") + 2:]
WORK, DATA, OUT = argv[0], argv[1], argv[2]
PRESET = argv[3] if len(argv) > 3 else mk_testpresets.MAIN_PRESET
WEIGHT = float(argv[4]) if len(argv) > 4 else 1.0
sys.argv = [sys.argv[0], "--", WORK, DATA, OUT]
import clip_test as ct  # noqa: E402  (reuses the project loader; its main() is not run on import below)


def material(name, image_path=None, color=(0.8, 0.65, 0.55, 1), alpha=False):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes["Principled BSDF"]
    if image_path:
        tex = m.node_tree.nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(image_path, check_existing=True)
        m.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
        if alpha:
            tex.image.alpha_mode = "STRAIGHT"
            m.node_tree.links.new(tex.outputs["Alpha"], bsdf.inputs["Alpha"])
            if hasattr(m, "surface_render_method"):
                m.surface_render_method = "BLENDED"
            else:
                m.blend_method = "BLEND"
            m.use_backface_culling = False
    else:
        bsdf.inputs["Base Color"].default_value = color
    return m


def add(name, v, t, uv, mat):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(map(float, p)) for p in v], [], [tuple(map(int, f)) for f in t])
    me.update()
    layer = me.uv_layers.new(name="UVMap")
    loops = np.zeros(len(me.loops), dtype=np.int64)
    me.loops.foreach_get("vertex_index", loops)
    c = np.asarray(uv, dtype=np.float64)[loops].copy()
    c[:, 1] = 1.0 - c[:, 1]
    layer.data.foreach_set("uv", c.ravel().astype(np.float32))
    me.shade_smooth()
    me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def main():
    presets = pickle.load(open(os.path.join(WORK, "ref", "presets.pkl"), "rb"))
    body = mk_fit.Body(pickle.load(open(os.path.join(WORK, "ref", "body.pkl"), "rb")))
    feet = mk_fit.Body(pickle.load(open(os.path.join(WORK, "ref", "feet.pkl"), "rb")), shape_names=("Feet",))
    attrs = list(body.slider_attrs) + [a for a in feet.slider_attrs if a["name"] not in body.slider_names]
    if PRESET.endswith(".json"):
        sys.path.insert(0, os.path.join(TOOLS, "dev"))
        from diag_wrap_values import player_values
        vals = player_values(PRESET, WEIGHT)
    else:
        vals = mk_preset.slider_values(attrs, presets.get(PRESET, {}), WEIGHT) if PRESET != "none" else {}
    atlas = os.path.join(WORK, "tex", "SunDancer_d.png")
    m_out = material("outfit", atlas)
    m_veil = material("veil", atlas, alpha=True)
    m_skin = material("skin", None, (0.78, 0.6, 0.5, 1))
    sd = os.path.join(DATA, "CalienteTools", "BodySlide", "ShapeData", "Sun Dancer")
    ct.SD = sd
    for stem in ("Wrap", LOWER, "Sandals", "Bracelets", "Armlets", "Veils", "Collar", "Navel"):
        if stem == "none" or not os.path.exists(os.path.join(sd, stem + ".nif")):
            continue
        nif = mk_nif.NifFile(os.path.join(sd, stem + ".nif"))
        shapes = ct.load_project(stem)
        for s in nif.shapes:
            sh = mk_nif.read_shape(s)
            if sh["name"].startswith("EV_"):      # invisible physics shapes
                continue
            v = shapes[sh["name"]].morphed(vals)
            skin = sh["name"] in ("BaseShape", "Feet")
            add(stem + sh["name"], v, sh["t"], sh["uv"],
                m_skin if skin else (m_veil if sh["alpha"] is not None else m_out))
    nif = mk_nif.NifFile(os.path.join(DATA, "meshes", "!UBE", "Tinesh", "SunDancer", "circlet.nif"))
    for s in nif.shapes:
        sh = mk_nif.read_shape(s)
        add("circlet" + sh["name"], sh["v"], sh["t"], sh["uv"], m_out)
    hands = mk_fit.Body(pickle.load(open(os.path.join(WORK, "ref", "hands.pkl"), "rb")), shape_names=("Hands",))
    hv = hands.apply_preset(mk_preset.slider_values(hands.slider_attrs, presets.get(PRESET, {}), WEIGHT)
                            if PRESET != "none" else {})
    ob = mk_blend.mesh_object("hands", hv, hands.t)
    ob.data.materials.append(m_skin)
    hd = pickle.load(open(os.path.join(WORK, "ref", "head.pkl"), "rb"))["shape"]
    ob = mk_blend.mesh_object("head", hd["v"], hd["t"])
    ob.data.materials.append(m_skin)
    # scythe held upright beside the figure
    sc = mk_nif.NifFile(os.path.join(DATA, "meshes", "Tinesh", "SunDancer", "scythe.nif"))
    m_sc = material("scythe", os.path.join(WORK, "tex", "EclipseScythe_d.png"))
    for s in sc.shapes:
        sh = mk_nif.read_shape(s)
        v = sh["v"][:, [0, 2, 1]] * np.array([-1, 1, 1]) + np.array([34.0, 0.0, 62.0])
        add("scythe" + sh["name"], v, sh["t"], sh["uv"], m_sc)
    sc_ = bpy.context.scene
    sc_.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in [e.identifier for e in
                                                                        bpy.types.RenderSettings.bl_rna.properties[
                                                                            "engine"].enum_items] else "BLENDER_EEVEE"
    sc_.render.resolution_x = 1100
    sc_.render.resolution_y = 1400
    world = bpy.data.worlds.new("w")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.42, 0.4, 0.38, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.9
    sc_.world = world
    for rot, energy in (((0.9, 0.2, 0.6), 3.5), ((1.2, -0.3, -2.4), 1.6)):
        ld = bpy.data.lights.new("sun", "SUN")
        ld.energy = energy
        lo = bpy.data.objects.new("sun", ld)
        lo.rotation_euler = rot
        sc_.collection.objects.link(lo)
    for name, d, c, s in (("front", (0, -1, 0), (12, 0, 70), 160), ("q34", (0.65, -0.75, -0.1), (4, 0, 70), 140),
                          ("back", (0, 1, 0), (0, 0, 66), 150), ("torso", (0.3, -0.95, -0.05), (0, 0, 88), 48),
                          ("hips", (-0.35, -0.94, 0.05), (0, 0, 64), 42), ("backup", (0.15, 1, -0.05), (0, 0, 92), 40),
                          ("backlow", (-0.2, 1, 0.1), (0, 0, 70), 36), ("legs", (0.5, -0.86, 0.0), (0, 0, 46), 60),
                          ("navel", (0.1, -1, 0.05), (0, 4, 78), 14), ("feet", (0.45, -0.9, -0.2), (0, 0, 10), 30),
                          ("scythe", (0, -1, 0), (40, 0, 75), 150),
                          ("fringe", (0, -1, -0.12), (0, 7, 93.5), 17), ("fringeside", (-1, -0.15, 0.05), (6, 7, 94), 14),
                          ("wristL", (0.75, -0.62, 0.15), (-26.3, -1.0, 78.0), 9),
                          ("wristR", (-0.75, -0.62, 0.15), (26.3, -1.0, 78.0), 9),
                          ("veilL", (1, 0.05, 0), (-19, -3, 90), 38), ("veilback", (0.25, 1, 0), (-14, -3, 88), 40),
                          ("arms", (0, -1, 0), (0, 0, 90), 64),
                          ("collar", (0.2, -1, -0.25), (0, 3, 104), 17), ("collarback", (0, 1, -0.15), (0, -4, 106), 17),
                          ("beltback", (0, 1, 0.05), (0, -8, 71), 13)):
        if VIEWS and name not in VIEWS:
            continue
        cam = mk_blend.camera(c, d, s)
        cam.data.ortho_scale = s
        sc_.render.resolution_x, sc_.render.resolution_y = (1100, 1400) if s > 100 else (1200, 1200)
        sc_.render.filepath = "%s_%s.png" % (OUT, name)
        bpy.ops.render.render(write_still=True)
        bpy.data.objects.remove(cam)


main()
