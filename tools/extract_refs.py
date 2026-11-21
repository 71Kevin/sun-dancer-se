"""Reads the UBE 2.0 reference meshes and BodySlide data from a Mod Organizer 2 mods folder (runs inside Blender).

  blender --background --factory-startup --python tools/extract_refs.py -- <out_dir>

Writes to <out_dir>:
  skeleton.pkl  global transforms of the XPMSSE female skeleton
  body.pkl      UBE body ("BaseShape") with every slider of the body slider set (with the Necoco add-on's sliders
                when that add-on is installed; the public packages leave them out, build_meshes.addon_sliders)
  feet.pkl      UBE feet ("Feet") with the feet slider set (likewise)
  hands.pkl     UBE hands ("Hands") with the hands slider set (likewise)
  head.pkl      UBE female head with its expression (FemaleHead.tri) and chargen (FemaleHeadCharGen.tri) morphs
  presets.pkl   the body presets the outfit is tested on (name -> {slider: {small, big}}): UBE's zeroed sliders and
                the personal presets of tools/presets.local.json (see tools/lib/mk_testpresets.py)
"""
import os
import pickle
import sys

import numpy as np

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(TOOLS, "lib"))
import mk_bs  # noqa: E402
import mk_nif  # noqa: E402
import mk_preset  # noqa: E402
import mk_testpresets  # noqa: E402
from mk_preset import MODS  # noqa: E402

UBE = os.path.join(MODS, "[NoDelete] UBE 2.0 - Ultimate Body Enhancer")
# optional: a community add-on with extra UBE sliders (shared on Discord); used when installed, never redistributed
NECOCO = os.path.join(MODS, "[NoDelete] UBE 2.0 - Necoco Body v2 (Discord)")
SKELETON = os.path.join(MODS, r"XP32 Maximum Skeleton Special Extended\meshes\actors\character"
                              r"\character assets female\skeleton_female.nif")
BS = r"CalienteTools\BodySlide"


def slider_set(part):
    """(slider set name, osp file, shape data folder, source nif) of a UBE part: the Necoco add-on's set when that
    add-on is installed (UBE's sliders plus the add-on's), else UBE's own set."""
    base = "UBE SE 2.0 Release %s" % part
    addon = os.path.join(NECOCO, BS, "SliderSets", base + " - Necoco Add-on.osp")
    if os.path.exists(addon):
        return base + " - Necoco Add-on", addon, base, base + ".nif"
    return base, os.path.join(UBE, BS, "SliderSets", base + ".osp"), base, base + ".nif"


SETS = {key: slider_set(part) for key, part in (("body", "Body"), ("feet", "Feet"), ("hands", "Hands"))}
HEAD_NIF = os.path.join(UBE, r"meshes\!UBE\Head\FemaleHead_tangent.nif")
HEAD_TRI = os.path.join(UBE, r"meshes\!UBE\Head\FemaleHead.tri")
HEAD_CHARGEN = os.path.join(UBE, r"meshes\!UBE\Head\FemaleHeadCharGen.tri")
PRESET_FILES = [(os.path.join(MODS, path), names) for path, names in mk_testpresets.PRESET_FILES] + [
    (os.path.join(UBE, r"CalienteTools\Bodyslide\SliderPresets\-Zeroed Sliders-.xml"), None),
]


def find_osd(folder, fname):
    for root in (UBE, NECOCO):
        p = os.path.join(root, BS, "ShapeData", folder, fname)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(fname)


def read_frtri(path):
    """FaceGen TRI (FRTRI003): base vertices and the differential morphs (int16 deltas * scale)."""
    import struct
    b = open(path, "rb").read()
    assert b[:8] == b"FRTRI003", path
    (nv, nt, nq, nl, nlt, nuv, ext, nmd, nms, nvadd) = struct.unpack_from("<10I", b, 8)
    off = 64
    v = np.frombuffer(b, "<f4", nv * 3, off).reshape(nv, 3).astype(np.float64)
    off += (nv + nvadd) * 12 + nt * 12 + nq * 16
    if ext & 1:
        off += nuv * 8 + nt * 12
    morphs = {}
    for _ in range(nmd):
        n, = struct.unpack_from("<I", b, off)
        off += 4
        name = b[off:off + n].rstrip(b"\x00").decode("latin-1")
        off += n
        scale, = struct.unpack_from("<f", b, off)
        off += 4
        d = np.frombuffer(b, "<i2", nv * 3, off).reshape(nv, 3).astype(np.float64) * scale
        off += nv * 6
        morphs[name] = d
    return v, morphs


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    skel = mk_nif.skeleton_globals(SKELETON)
    pickle.dump(skel, open(os.path.join(out_dir, "skeleton.pkl"), "wb"))
    print("skeleton nodes", len(skel))
    for key, (set_name, osp, folder, source) in SETS.items():
        sliders = mk_bs.read_slider_set(osp, set_name)
        nif = mk_nif.NifFile(os.path.join(UBE, BS, "ShapeData", folder, source))
        shapes = {s.name: mk_nif.read_shape(s) for s in nif.shapes}
        osd_cache, diffs = {}, {}
        for attrs, data in sliders:
            for dname, target, dfile in data:
                fname, entry = dfile.split("\\", 1)
                if fname not in osd_cache:
                    osd_cache[fname] = mk_bs.read_osd(find_osd(folder, fname))
                diffs[(attrs["name"], target)] = osd_cache[fname].get(entry, (np.zeros(0, np.int64), np.zeros((0, 3))))
        for name, sh in shapes.items():
            dev = [np.abs(np.linalg.inv(sh["s2b"][j]) - skel[b]).max() for j, b in enumerate(sh["bones"]) if b in skel]
            print(key, name, len(sh["v"]), "verts", len(sh["bones"]), "bones, max bind deviation from skeleton %.4f"
                  % (max(dev) if dev else -1))
        pickle.dump({"set": set_name, "sliders": sliders, "shapes": shapes, "diffs": diffs},
                    open(os.path.join(out_dir, key + ".pkl"), "wb"))
        print(key, set_name, len(sliders), "sliders,", sum(1 for d in diffs.values() if len(d[0])), "non-empty diffs")
    head = mk_nif.read_shape(mk_nif.NifFile(HEAD_NIF).shapes[0])
    tv, expr = read_frtri(HEAD_TRI)
    cv, chargen = read_frtri(HEAD_CHARGEN)
    print("head", len(head["v"]), "verts; tri", len(tv), "verts,", len(expr), "expressions;", len(chargen), "chargen",
          "max |nif - tri| %.4f" % np.abs(tv[:len(head["v"])] - head["v"]).max() if len(tv) >= len(head["v"]) else "")
    pickle.dump({"shape": head, "expressions": expr, "chargen": chargen}, open(os.path.join(out_dir, "head.pkl"), "wb"))
    presets = {}
    for path, names in PRESET_FILES:
        if not os.path.exists(path):      # the test presets are optional (only the QA tools use them)
            print("presets file not found, skipped:", path)
            continue
        import re
        text = open(path, encoding="utf-8-sig", errors="replace").read()
        for pname in re.findall(r'<Preset name="([^"]+)"', text):
            if names is None or pname in names:
                presets[pname] = mk_preset.read_preset(path, pname)[1]
    pickle.dump(presets, open(os.path.join(out_dir, "presets.pkl"), "wb"))
    print("presets:", ", ".join(presets))


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    main(argv[0])
