"""After PGPatcher: are the Sun Dancer meshes in its output patched for PBR, and did everything else survive (SMP
system string, physics bones, invisible colliders, BodySlide morph link, geometry)? Runs inside Blender:
  blender -b --factory-startup --python tools/dev/verify_pg.py -- [<mods_dir>] [--pg <mod>] [--bs <mod>] [--main <mod>]
Every Sun Dancer NIF the game would load (the BodySlide output overrides the main mod) is paired with the same path in the
PGPatcher output. Only shapes textured from textures\\Tinesh\\SunDancer are expected to turn PBR (skin shapes keep
the skin's textures). The invisible physics colliders (EV_*) carry the atlas too, so PGPatcher re-textures them for PBR
as well; that is harmless as long as they stay invisible: the alpha-blend property unchanged and the material alpha 0
(Community Shaders multiplies the final alpha by it on the PBR path too: `alpha *= MaterialData.z` in Lighting.hlsl),
no shadow casting and no depth writes."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import mk_nif  # noqa: E402

args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
MODS = args[0] if args and not args[0].startswith("--") else r"C:\Modding\TrueNorth\mods"


def opt(name, default):
    return args[args.index(name) + 1] if name in args else default


PG = os.path.join(MODS, opt("--pg", "[NoDelete] Tinesh - PG Patcher Output"))
BS = os.path.join(MODS, opt("--bs", "[NoDelete] Tinesh - BodySlide Output UBE"))
MAIN = os.path.join(MODS, opt("--main", "[NoDelete] Tinesh - Sun Dancer (UBE)"))
RELS = (r"meshes\!UBE\Tinesh\SunDancer", r"meshes\Tinesh\SunDancer")


def game_meshes():
    found = {}
    for root in (MAIN, BS):          # later wins, as in the mod list
        for rel in RELS:
            d = os.path.join(root, rel)
            if os.path.isdir(d):
                for f in sorted(os.listdir(d)):
                    if f.lower().endswith(".nif"):
                        found[os.path.join(rel, f).lower()] = os.path.join(d, f)
    return found


def summary(path):
    nif = mk_nif.NifFile(path)
    shapes = {s.name: mk_nif.read_shape(s) for s in nif.shapes}
    strings = [(e.name, e.string_data) for e in nif.root.extra_data() if e.blockname == "NiStringExtraData"]
    ev = {n: mk_nif.xf_to_mat(node.transform) for n, node in nif.nodes.items() if n.startswith("EV ")}
    return shapes, strings, ev


def ours(sh):
    return any("sundancer" in t.lower() for t in sh["textures"].values())


problems, unpatched, patched_shapes = [], [], 0
meshes = game_meshes()
for rel, ref_path in meshes.items():
    name = os.path.basename(rel)
    pg_path = os.path.join(PG, rel)
    if not os.path.exists(pg_path):
        rs, _, _ = summary(ref_path)
        if any(ours(s) for n, s in rs.items() if not n.startswith("EV_")):
            unpatched.append(name)
        else:
            print("%-24s not patched (skin only, expected)" % name)
        continue
    rs, rx, rev = summary(ref_path)
    ps, px, pev = summary(pg_path)
    if set(rs) != set(ps):
        problems.append("%s: shape set changed %s" % (name, sorted(set(rs) ^ set(ps))))
    pbr_here = 0
    for sn in sorted(set(rs) & set(ps)):
        a, b = rs[sn], ps[sn]
        if len(a["v"]) != len(b["v"]) or len(a["t"]) != len(b["t"]):
            problems.append("%s/%s: vertex/triangle count changed" % (name, sn))
            continue
        moved = float(np.abs(a["v"] - b["v"]).max()) if len(a["v"]) else 0.0
        if moved > 1e-4:
            problems.append("%s/%s: vertices moved (max %.4f)" % (name, sn, moved))
        for field in ("bones", "alpha", "extras", "partitions"):
            if a[field] != b[field]:
                problems.append("%s/%s: %s changed %s -> %s" % (name, sn, field, a[field], b[field]))
        if sn.startswith("EV_"):
            f1 = int(b["shader"].get("Shader_Flags_1", 0))
            f2 = int(b["shader"].get("Shader_Flags_2", 0))
            invisible = (b["alpha"] is not None and b["alpha"][0] & 1 and float(b["shader"].get("Alpha", 1)) == 0.0
                         and not f1 & 0x200 and not f2 & 0x1)   # blend on, alpha 0, no Cast_Shadows, no ZBuffer_Write
            if not invisible:
                problems.append("%s/%s: the physics collider may be VISIBLE (alpha %s, material alpha %s, f1 %s, f2 %s)"
                                % (name, sn, b["alpha"], b["shader"].get("Alpha"), hex(f1), hex(f2)))
            continue
        if not ours(a):
            continue
        if any("pbr" in t.lower() for t in b["textures"].values()):
            pbr_here += 1
        else:
            problems.append("%s/%s: no PBR texture (diffuse %s)" % (name, sn, b["textures"].get("Diffuse", "")))
    patched_shapes += pbr_here
    if rx != px:
        problems.append("%s: root strings changed %s -> %s" % (name, rx, px))
    if set(rev) != set(pev):
        problems.append("%s: physics bones changed (%d -> %d)" % (name, len(rev), len(pev)))
    elif rev:
        worst = max(float(np.abs(rev[n] - pev[n]).max()) for n in rev)
        if worst > 1e-5:
            problems.append("%s: physics bone transforms changed (max %.5f)" % (name, worst))
    print("%-24s PBR shapes %d  root strings %s  physics bones %d"
          % (name, pbr_here, [k for k, _ in px], len(pev)))
print("Sun Dancer meshes the game loads: %d | patched by PGPatcher: %d | PBR shapes: %d"
      % (len(meshes), sum(os.path.exists(os.path.join(PG, r)) for r in meshes), patched_shapes))
if unpatched:
    problems.append("not in the PGPatcher output (no PBR): " + ", ".join(unpatched))
print("PROBLEMS", len(problems))
for p in problems:
    print("  ", p)
