"""Compares BodySlide's output with the projects' ShapeData NIFs (runs inside Blender):
  blender -b --factory-startup --python tools/dev/verify_bodyslide.py -- [<shapedata_dir> <output_dir> <xml_dir>]
Checks that every shape, vertex/triangle count, bone list, alpha and texture set survived, that the SMP system string,
the physics bones (with their rest transforms) and every bone/shape the system files name are still there, and prints
how far each shape moves between the weight-0 and weight-100 builds and from its source."""
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import mk_nif  # noqa: E402

MODS = r"C:\Modding\TrueNorth\mods"
MAIN = os.path.join(MODS, "[NoDelete] Tinesh - Sun Dancer (UBE)")
args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
src_dir, out_dir, xml_dir = args[:3] if len(args) >= 3 else (
    os.path.join(MAIN, r"CalienteTools\BodySlide\ShapeData\Sun Dancer"),
    os.path.join(MODS, r"[NoDelete] Tinesh - BodySlide Output UBE\meshes\!UBE\Tinesh\SunDancer"),
    os.path.join(MAIN, r"meshes\!UBE\Tinesh\SunDancer"))
PROJECTS = ["Wrap", "Loincloth", "LoinclothNoBriefs", "Briefs", "Sandals", "Bracelets", "Armlets", "Veils", "Collar",
            "Navel"]
problems = []


def load(path):
    nif = mk_nif.NifFile(path)
    return nif, {s.name: mk_nif.read_shape(s) for s in nif.shapes}


def root_strings(nif):
    return [(ed.name, ed.string_data) for ed in nif.root.extra_data() if ed.blockname == "NiStringExtraData"]


for proj in PROJECTS:
    src_nif, src = load(os.path.join(src_dir, proj + ".nif"))
    outs = {k: load(os.path.join(out_dir, "%s_%d.nif" % (proj.lower(), k))) for k in (0, 1)}
    print("==", proj, "shapes src", len(src), "out", len(outs[0][1]), len(outs[1][1]))
    for k, (nif, sh) in outs.items():
        if set(sh) != set(src):
            problems.append("%s_%d shape set differs: %s" % (proj, k, sorted(set(sh) ^ set(src))))
        for name in set(sh) & set(src):
            s, r = sh[name], src[name]
            if len(s["v"]) != len(r["v"]) or len(s["t"]) != len(r["t"]):
                problems.append("%s_%d %s vertex/triangle count differs" % (proj, k, name))
            for field in ("bones", "alpha", "textures"):
                if s[field] != r[field]:
                    problems.append("%s_%d %s %s differ" % (proj, k, name, field))
        if root_strings(nif) != root_strings(src_nif):
            problems.append("%s_%d root strings differ" % (proj, k))
        ev = sorted(n for n in nif.nodes if n.startswith("EV "))
        if ev != sorted(n for n in src_nif.nodes if n.startswith("EV ")):
            problems.append("%s_%d physics bones differ" % (proj, k))
    for name in sorted(outs[1][1]):
        a, b = outs[0][1][name]["v"], outs[1][1][name]["v"]
        d01 = float(np.linalg.norm(a - b, axis=1).max()) if len(a) == len(b) else -1
        ds = (float(np.linalg.norm(b - src[name]["v"], axis=1).max())
              if name in src and len(b) == len(src[name]["v"]) else -1)
        print("   %-22s v=%6d  |w0-w100| max %.3f  |w100-src| max %.3f  alpha=%s"
              % (name, len(b), d01, ds, outs[1][1][name]["alpha"]))
    nif = outs[1][0]
    strings = root_strings(nif)
    if strings:
        print("   root strings", strings)
    ev = sorted(n for n in nif.nodes if n.startswith("EV "))
    if ev:
        worst = max(float(np.abs(mk_nif.xf_to_mat(nif.nodes[n].transform)
                                 - mk_nif.xf_to_mat(src_nif.nodes[n].transform)).max()) for n in ev)
        print("   physics bones %d, rest transform max diff vs source %.5f" % (len(ev), worst))
    for key, value in strings:
        if "Physics" not in key:
            continue
        text = open(os.path.join(xml_dir, os.path.basename(value)), encoding="utf-8").read()
        bones = set(re.findall(r'<bone name="([^"]+)"', text))
        shapes = set(re.findall(r'<per-(?:vertex|triangle)-shape name="([^"]+)"', text))
        miss_b = sorted(b for b in bones if b not in nif.nodes)
        miss_s = sorted(s for s in shapes if s not in outs[1][1])
        print("   %s: %d bones, shapes %s; missing bones %s, missing shapes %s"
              % (os.path.basename(value), len(bones), sorted(shapes), miss_b, miss_s))
        if miss_b or miss_s:
            problems.append("%s: the system file names bones/shapes the NIF lacks" % proj)
print("PROBLEMS", len(problems))
for p in problems:
    print("  ", p)
