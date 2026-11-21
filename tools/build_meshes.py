"""Builds the Sun Dancer meshes, BodySlide projects, ground models and the texture manifest (runs in Blender).

  blender --background --factory-startup --python tools/build_meshes.py -- <work> <out> [piece,piece,...]

<work> holds the reference data (tools/extract_refs.py) and the caches; <out>\\data receives the mod files and
<out>\\meshes_report.json the bounds, slider counts and collision data the plugin generator reads."""
import json
import os
import pickle
import sys
import time

import numpy as np

TOOLS = os.path.dirname(os.path.abspath(__file__))
for sub in ("lib", "pieces", ""):
    sys.path.insert(0, os.path.join(TOOLS, sub))

import mk_atlas  # noqa: E402
import mk_blend  # noqa: E402
import mk_bs  # noqa: E402
import mk_export  # noqa: E402
import mk_fit  # noqa: E402
import mk_geom  # noqa: E402
import mk_preset  # noqa: E402
import mk_rigid as rigid  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
WORK = argv[0]
OUT = argv[1]
ONLY = argv[2].split(",") if len(argv) > 2 and argv[2] else None
REF = os.path.join(WORK, "ref")
VANILLA = os.path.join(WORK, "refs", "vanilla", "meshes")
MESH_REL = "!UBE\\Tinesh\\SunDancer"
MESH_DIR = os.path.join(OUT, "data", "meshes", "!UBE", "Tinesh", "SunDancer")
BS_DIR = os.path.join(OUT, "data", "CalienteTools", "BodySlide")
SHAPEDATA = os.path.join(BS_DIR, "ShapeData", "Sun Dancer")
ATLAS = 8192
GROUP = "Sun Dancer"
GND_TEMPLATE = os.path.join(VANILLA, "armor", "studded", "male", "body_go.nif")

# key, display stem, file name, partition (slot), BodySlide project (body-following pieces)
ITEMS = [
    ("wrap", "Wrap", "wrap", 32, True),
    ("loincloth", "Loincloth", "loincloth", 52, True),       # with the briefs (the original look)
    ("loinclothnobriefs", "LoinclothNoBriefs", "loinclothnobriefs", 52, True),
    ("briefs", "Briefs", "briefs", 52, True),
    ("sandals", "Sandals", "sandals", 37, True),
    ("bracelets", "Bracelets", "bracelets", 34, True),
    ("armlets", "Armlets", "armlets", 58, True),
    ("veils", "Veils", "veils", 59, True),
    ("collar", "Collar", "collar", 45, True),
    ("circlet", "Circlet", "circlet", 42, False),
    ("navel", "Navel", "navel", 49, True),
]
# items not built by piece_<key>.build: (module, function)
BUILDERS = {"loincloth": ("piece_loincloth", "build_with_briefs"), "loinclothnobriefs": ("piece_loincloth", "build")}
# BodySlide project names that differ from "Sun Dancer - <stem>"
TITLES = {"LoinclothNoBriefs": "Loincloth without Briefs"}
SMP_KEY = "HDT Skinned Mesh Physics Object"
t0 = time.time()


def log(msg):
    print("[%6.1fs] %s" % (time.time() - t0, msg), flush=True)


def shader_for(group):
    mats = {p.mat for p in group}
    if "collision" in mats:
        return "invisible"
    if "chiffon" in mats:
        return "chiffon"
    if mats & {"gold", "gem", "steel"} and not mats & {"linen", "leather", "lining", "edge"}:
        return "gold"
    return "cloth"


def group_by_shape(parts):
    order = []
    for p in parts:
        if p.shape not in order:
            order.append(p.shape)
    return [(name, [p for p in parts if p.shape == name]) for name in order]


def lay_down(v):
    r = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
    p = v @ r.T
    c = 0.5 * (p.min(0) + p.max(0))
    return p, r, c


def write_gnd(path, parts, mass, root_name):
    """Ground model: the item laid on its back, static shapes and a convex collision (template: vanilla body_go)."""
    from io_scene_nifly.pyn.pynifly import NifFile
    base = NifFile(GND_TEMPLATE)
    keep = [p for p in parts if p.mat != "collision"]
    allv = np.concatenate([p.v for p in keep])
    _, r, _ = lay_down(allv)
    lv = allv @ r.T
    c = 0.5 * (lv.min(0) + lv.max(0))
    zmin = lv[:, 2].min()
    nif = NifFile()
    nif.initialize("SKYRIMSE", path, root_type="BSFadeNode", root_name=root_name)
    root = nif.root
    rigid.copy_root_extra_data(base.root, nif, root)
    groups = []
    for name, group in group_by_shape(keep):
        data = mk_export.merge_parts(group)
        p = data["v"] @ r.T
        p[:, 0] -= c[0]
        p[:, 1] -= c[1]
        p[:, 2] -= zmin
        key = shader_for(group)
        groups.append((name, {"v": p, "t": data["t"], "uv": data["uv"], "n": data["n"] @ r.T}, key,
                       mk_export.CHIFFON_ALPHA if key == "chiffon" else None))
    mk_export.static_shapes(nif, groups, parent=root)
    pts = np.concatenate([g[1]["v"] for g in groups])
    cal = rigid.inertia_calibration(base.root.collision_object.body)
    info = rigid.add_convex_collision(root, pts, base.root.collision_object, mass, cal, 48)
    nif.save()
    return {"collision": info, "bounds": rigid.bounds(pts)}


def hemisphere(count, seed=7):
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(count * 3, 3))
    d = d[d[:, 2] > 0.15][:count]
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def compute_ao(parts, occluders, rays=14, reach=1.8):
    from mathutils import Vector
    vs, ts = [], []
    off = 0
    for v, t in occluders:
        vs.append(v)
        ts.append(t + off)
        off += len(v)
    tree = mk_blend.bvh(np.concatenate(vs), np.concatenate(ts))
    dirs = hemisphere(rays)
    for p in parts:
        n = p.normals if getattr(p, "normals", None) is not None else mk_geom.vertex_normals(p.v, p.t)
        a = np.where(np.abs(n[:, 2:3]) < 0.9, np.array([[0, 0, 1.0]]), np.array([[1.0, 0, 0]]))
        tx = mk_geom.normalize(np.cross(n, a))
        ty = np.cross(n, tx)
        occ = np.zeros(len(p.v))
        for d in dirs:
            w = tx * d[0] + ty * d[1] + n * d[2]
            o = p.v + n * 0.03
            for i in range(len(o)):
                h = tree.ray_cast(Vector(o[i]), Vector(w[i]), reach)
                if h[0] is not None:
                    occ[i] += (1.0 - h[3] / reach) ** 0.5
        p.attrs["ao"] = 1.0 - occ / len(dirs)


def compute_worn_ao(built, skin_v, skin_t):
    """Ambient occlusion baked into the atlas, each part shaded only by what is always worn with it: its own item,
    the items of the other slots and the skin (the sheer veils shade nothing). Items of one slot replace each other
    (the Loincloth with Briefs, the Loincloth without Briefs and the Briefs, slot 52), and a part shared by several
    items is shaded only by what they all have: the briefs by the briefs, not by the loincloth's panels and belt.
    Shaded by every item at once, the Briefs worn alone showed the back panel's edges and the belt as brown smudges
    on the seat (round 6)."""
    slot = {key: partition for key, _, _, partition, _ in ITEMS}

    def shades(p):
        return p.mat not in ("collision", "chiffon") and not p.island.startswith("patch:")
    worn_with = {k: {id(q) for kk, ps in built.items() if kk == k or slot[kk] != slot[k] for q in ps if shades(q)}
                 for k in built}
    owners, every = {}, {}
    for key, parts in built.items():
        for p in parts:
            every[id(p)] = p
            owners.setdefault(id(p), set()).add(key)
    groups = {}
    for pid, keys in owners.items():
        p = every[pid]
        if p.mat == "collision" or p.island.startswith("patch:"):
            continue
        occ = frozenset.intersection(*(frozenset(worn_with[k]) for k in keys))
        groups.setdefault(occ, []).append(p)
    for occ, parts in groups.items():
        compute_ao(parts, [(every[i].v, every[i].t) for i in sorted(occ)] + [(skin_v, skin_t)])
    log("ambient occlusion: %d occluder sets (%s)" % (len(groups), ", ".join(
        "%d parts / %d occluders" % (len(ps), len(o)) for o, ps in groups.items())))


def manifest_entry(p):
    return {"name": p.name, "mat": p.mat, "island": p.island, "v": p.v.astype(np.float32), "t": p.t.astype(np.int32),
            "uv": p.uv_px.astype(np.float32),
            "n": (p.normals if getattr(p, "normals", None) is not None
                  else mk_geom.vertex_normals(p.v, p.t)).astype(np.float32),
            "attrs": {k: np.asarray(a, dtype=np.float32) for k, a in p.attrs.items()
                      if np.asarray(a).ndim == 1 and len(a) == len(p.v)}}


PULSE = [(0.0, 1.0), (0.6, 0.86), (1.3, 0.97), (2.0, 0.8), (2.7, 0.95), (3.4, 1.0)]
SCYTHE_STEM = "EclipseScythe"


def scythe_shader(group):
    mats = {p.mat for p in group}
    if "ember" in mats:
        return mk_export.shader("glow", SCYTHE_STEM)
    if mats & {"gold"}:
        return mk_export.shader("gold", SCYTHE_STEM)
    if mats & {"steel", "obsidian"}:
        return mk_export.shader("steel", SCYTHE_STEM)
    return mk_export.shader("cloth", SCYTHE_STEM)


def write_weapon(path, parts, template, collision_template, root_name, mass):
    from io_scene_nifly.pyn.pynifly import NifFile
    base = NifFile(template)
    coll = NifFile(collision_template)
    bsx = [int(ed.properties.integerData) for ed in base.root.extra_data() if ed.blockname == "BSXFlags"][0]
    nif = NifFile()
    nif.initialize("SKYRIMSE", path, root_type="BSFadeNode", root_name=root_name)
    root = nif.root
    rigid.copy_root_extra_data(base.root, nif, root, bsx_flags=bsx | 1)
    groups = []
    for name, group in group_by_shape(parts):
        data = mk_export.merge_parts(group)
        groups.append((name, {"v": data["v"], "t": data["t"], "uv": data["uv"], "n": data["n"]}, scythe_shader(group),
                       None))
    shapes = mk_export.static_shapes(nif, groups, parent=root)
    glow = [s for s, g in zip(shapes, groups) if g[2][0]["Shader_Type"] == 2]
    if glow:
        rigid.add_emissive_loop(nif, glow[0], [(t, (1.0 * f, 0.72 * f, 0.3 * f)) for t, f in PULSE])
    pts = np.concatenate([g[1]["v"] for g in groups])
    cal = rigid.inertia_calibration(coll.root.collision_object.body)
    info = rigid.add_convex_collision(root, pts, coll.root.collision_object, mass, cal, 48)
    nif.save()
    return {"collision": info, "bounds": rigid.bounds(pts)}


def build_scythe(report):
    import piece_scythe
    parts = piece_scythe.build(log=log)["parts"]
    layout = mk_atlas.pack(parts, ATLAS)
    mk_atlas.apply(parts, layout)
    steel = os.path.join(VANILLA, "weapons", "steel")
    report["items"]["scythe"] = write_weapon(os.path.join(MESH_DIR, "..", "..", "..", "Tinesh", "SunDancer",
                                                          "scythe.nif"), parts,
                                             os.path.join(steel, "steelbattleaxe.nif"),
                                             os.path.join(steel, "1stpersonsteelwaraxe.nif"), "EclipseScythe", 10.0)
    compute_ao([p for p in parts if not p.island.startswith("patch:")], [(p.v, p.t) for p in parts])
    manifest = {"size": ATLAS, "layout": layout, "parts": [manifest_entry(p) for p in parts], "stem": SCYTHE_STEM}
    with open(os.path.join(WORK, "texture_manifest_scythe.pkl"), "wb") as f:
        pickle.dump(manifest, f)
    log("scythe exported, atlas %.1f px/unit" % layout["scale"])


ARM_KEYS = ("upperarm", "forearm", "hand", "finger")


def write_first_person(bdata, bsh, globals_, limit=0.5):
    """First-person model of the Wrap: only the arms of its UBE skin (triangles whose three vertices are weighted at
    least `limit` to arm bones), as _0 and _1 for the weight slider. The garment itself is never in view."""
    bones = bdata["bones"]
    arm = np.array([any(k in b.lower() for k in ARM_KEYS) and "clavicle" not in b.lower() for b in bones])
    score = (bdata["w"][:, arm]).sum(1)
    keep = (score[bdata["t"]] >= limit).all(1)
    t = bdata["t"][keep]
    used = np.unique(t)
    remap = -np.ones(len(bdata["v"]), dtype=np.int64)
    remap[used] = np.arange(len(used))
    data = {k: bdata[k][used] for k in ("v", "uv", "n", "w")}
    data.update(t=remap[t], bones=bones, sliders=[], D=np.zeros((len(used), 0, 3), dtype=np.float32))
    ns = mk_export.NifSet("Wrap1st", "wrap_1st", 32, globals_)
    ns.add_shape("BaseShape", data, shader=bsh["shader"], textures=bsh["textures"])
    for suffix in ("_0", "_1"):
        ns.write_single(os.path.join(MESH_DIR, "wrap_1st%s.nif" % suffix))
    log("first-person arms: %d triangles" % len(t))


def load_body():
    import piece_wrap
    ref = pickle.load(open(os.path.join(REF, "body.pkl"), "rb"))
    body = mk_fit.Body(ref)
    path = os.path.join(WORK, "proxy.pkl")
    if os.path.exists(path):
        proxy = pickle.load(open(path, "rb"))
        proxy.tree = mk_blend.bvh(proxy.v, proxy.t)
    else:
        proxy = piece_wrap.make_proxy(body)
        tree, proxy.tree = proxy.tree, None
        pickle.dump(proxy, open(path, "wb"))
        proxy.tree = tree
    return ref, body, proxy


def bone_globals(skel, *bodies):
    g = dict(skel)
    for b in bodies:
        for j, name in enumerate(b.bones):
            g[name] = np.linalg.inv(b.s2b[j])
    return g


def addon_sliders():
    """Sliders of an add-on to the UBE slider sets (body, feet, hands): in an add-on set the UBE sliders keep their data
    in UBE's own .osd, while the add-on's sliders take theirs from the add-on's .osd, named after the set. Their data
    belong to the add-on's author, so the public packages leave them out (MK_PUBLIC=1, build.ps1 -Public)."""
    out = set()
    for key in ("body", "feet", "hands"):
        ref = pickle.load(open(os.path.join(REF, key + ".pkl"), "rb"))
        own = ref["set"] + ".osd"
        files = [{d[2].split("\\")[0] for d in data} for _, data in ref["sliders"]]
        if not any(f and own not in f for f in files):
            continue        # a plain UBE set, no add-on
        out |= {attrs["name"] for (attrs, _), f in zip(ref["sliders"], files) if own in f}
    return out


def main():
    import importlib
    if os.environ.get("MK_PUBLIC") == "1":
        mk_export.EXCLUDE_SLIDERS = addon_sliders()
        log("public build: %d add-on sliders left out of the BodySlide data" % len(mk_export.EXCLUDE_SLIDERS))
    if ONLY == ["scythe"]:
        os.makedirs(os.path.join(OUT, "data", "meshes", "Tinesh", "SunDancer"), exist_ok=True)
        report = {"items": {}}
        build_scythe(report)
        merge_report(report)
        return
    ref, body, proxy = load_body()
    skel = pickle.load(open(os.path.join(REF, "skeleton.pkl"), "rb"))
    corpus = mk_preset.corpus(body.slider_attrs) + mk_preset.extra_samples(WORK)
    log("references, %d preset samples" % len(corpus))
    built, extra = {}, {}
    for key, stem, out_name, partition, _ in ITEMS:
        if ONLY and key not in ONLY:
            continue
        module, func = BUILDERS.get(key, ("piece_" + key, "build"))
        fn = getattr(importlib.import_module(module), func)
        res = fn(body, proxy, log, corpus=corpus) if key == "wrap" else fn(body, proxy, log)
        built[key] = res["parts"]
        extra[key] = res
        log("%s built" % key)
    every = [p for parts in built.values() for p in parts]
    layout = mk_atlas.pack(every, ATLAS)
    mk_atlas.apply(every, layout)
    log("atlas %.1f px/unit, %d islands" % (layout["scale"], len(layout["islands"])))
    others = [extra[k]["ref_body"] for k in extra if extra[k].get("ref_body") is not None]
    globals_ = bone_globals(skel, body, *others)
    for res in extra.values():
        if res.get("fx"):
            globals_.update(res["fx"]["globals"])     # physics bones (cloth chains) and their skeleton chain
    os.makedirs(MESH_DIR, exist_ok=True)
    os.makedirs(SHAPEDATA, exist_ok=True)
    report = {"atlas": {"size": ATLAS, "px_per_unit": layout["scale"]}, "items": {}}
    sets, names = [], []
    for key, stem, out_name, partition, bodyslide in ITEMS:
        if key not in built:
            continue
        res = extra[key]
        ns = mk_export.NifSet(stem, out_name, partition, globals_)
        if res.get("fx"):
            # SMP: the chain bones as nodes under the pelvis, the system file named on the root
            ns.nodes = [(n, parent, globals_[n]) for n, parent in res["fx"]["hierarchy"]]
            ns.root_extras.append((SMP_KEY, "meshes\\%s\\%s.xml" % (MESH_REL, out_name)))
            with open(os.path.join(MESH_DIR, out_name + ".xml"), "w", encoding="utf-8", newline="\r\n") as f:
                f.write(res["fx"]["xml"])
        slider_attrs = list(body.slider_attrs)
        if key == "wrap":
            bdata, bsh = mk_export.body_shape_data(ref, body, res["keep_tris"], D_weld=res["D_conf"],
                                                   changed=res["under"])
            # all of the kept skin in partition 32: the UBE body puts the pubic area up to the lower belly in
            # partition 53, which the game hides whenever another item uses slot 53 (the player's chest piercings
            # do: the strip of belly between the briefs and z 73.5 vanished and showed the background)
            ns.add_shape("BaseShape", bdata, shader=bsh["shader"], textures=bsh["textures"], keep_all=True,
                         partition=32)
            slider_attrs += list(body.uv_attrs)
        if key == "wrap":
            write_first_person(bdata, bsh, globals_)
        if res.get("skin"):
            sdata, ssh, sattrs = res["skin"]
            ns.add_shape(sdata["name"], sdata, shader=ssh["shader"], textures=ssh["textures"], keep_all=True)
            have = {a["name"] for a in slider_attrs}
            slider_attrs += [a for a in sattrs if a["name"] not in have]
        for name, group in group_by_shape(built[key]):
            bones = res.get("bones", body.bones)
            data = mk_export.merge_parts(group, bones)
            data["w"], data["D"] = mk_fit.weld_attributes(data["v"], [data["w"], data["D"]])
            ns.add_shape(name, data, shader_for(group), extras=res.get("extras", {}).get(name, ()))
        if bodyslide:
            ns.write_shapedata(SHAPEDATA)
            ns.write_meshes(MESH_DIR, "%s\\%s.tri" % (MESH_REL, out_name), slider_attrs)
            title = "Sun Dancer - %s" % TITLES.get(stem, stem)
            sets.append(ns.slider_set(title, GROUP, "meshes\\" + MESH_REL, slider_attrs))
            names.append(title)
        else:
            ns.write_single(os.path.join(MESH_DIR, "%s.nif" % out_name))
        allv = np.concatenate([s["data"]["v"] for s in ns.shapes if s["name"] not in ("BaseShape", "Feet")])
        report["items"][key] = {"shapes": {s["name"]: [len(s["data"]["v"]), len(s["data"]["t"])] for s in ns.shapes},
                                "sliders": {s["name"]: len(s["diffs"]) for s in ns.shapes},
                                "bounds": rigid.bounds(allv)}
        report["items"][key]["gnd"] = write_gnd(os.path.join(MESH_DIR, "%s_gnd.nif" % out_name), built[key],
                                                1.0 if key not in ("wrap", "loincloth", "loinclothnobriefs") else 1.5,
                                                "SunDancer%sGnd" % stem)
        log("%s exported" % key)
    if sets:
        mk_bs.write_osp(os.path.join(BS_DIR, "SliderSets", "Sun Dancer.osp"), sets)
        mk_bs.write_groups(os.path.join(BS_DIR, "SliderGroups", "Sun Dancer.xml"),
                           {GROUP: names, "UBE": names, "UBE Outfits": names})
    # the skin as worn: with the Wrap, only its trimmed skin (the full body's nipples and vulva would darken the
    # cups and briefs from inside)
    skin_t = body.t if "wrap" not in extra else body.inv[body.orig_t["BaseShape"]][extra["wrap"]["keep_tris"]]
    compute_worn_ao(built, body.v, skin_t)
    log("ambient occlusion")
    manifest = {"size": ATLAS, "layout": layout, "parts": [manifest_entry(p) for p in every], "stem": mk_export.STEM}
    with open(os.path.join(WORK, "texture_manifest.pkl"), "wb") as f:
        pickle.dump(manifest, f)
    if not ONLY or "scythe" in ONLY:
        os.makedirs(os.path.join(OUT, "data", "meshes", "Tinesh", "SunDancer"), exist_ok=True)
        build_scythe(report)
    merge_report(report)
    log("done")


def merge_report(report):
    path = os.path.join(OUT, "meshes_report.json")
    old = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {"items": {}}
    old.setdefault("items", {}).update(report["items"])
    for k, v in report.items():
        if k != "items":
            old[k] = v
    with open(path, "w", encoding="utf-8") as f:
        json.dump(old, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


main()
