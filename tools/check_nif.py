"""Reads back built NIFs and reports what the game and BodySlide will see (runs inside Blender):
shapes, vertex/triangle counts, bones, partitions, shader type and textures, extra data, weight sanity.
  blender -b --factory-startup --python tools/check_nif.py -- <nif> [<nif> ...]"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
import mk_nif  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1:]
problems = 0
for path in argv:
    nif = mk_nif.NifFile(path)
    print("==", os.path.basename(path), "root", nif.rootName, "shapes", len(nif.shapes))
    for ed in nif.root.extra_data():
        print("   root extra", ed.blockname, getattr(ed, "name", ""), getattr(ed, "string_data", ""))
    for s in nif.shapes:
        sh = mk_nif.read_shape(s)
        w = sh["w"]
        wsum = w.sum(1) if w.size else np.ones(len(sh["v"]))
        maxb = (w > 0).sum(1).max() if w.size else 0
        bad = np.abs(wsum - 1) > 0.01 if w.size else np.zeros(len(sh["v"]), bool)
        print("   %-18s v=%6d t=%6d bones=%3d parts=%s shader=%d flags1=%s maxbones/vert=%d unweighted=%d"
              % (sh["name"], len(sh["v"]), len(sh["t"]), len(sh["bones"]), sh["partitions"],
                 int(sh["shader"].get("Shader_Type", -1)), hex(int(sh["shader"].get("Shader_Flags_1", 0))), maxb,
                 int(bad.sum())))
        print("      textures", {k: v for k, v in sh["textures"].items()})
        if sh["extras"]:
            print("      extras", [(n, (v[:60] + "...") if len(v) > 60 else v) for n, v in sh["extras"]])
        if bad.any() or len(sh["v"]) > 65535:
            problems += 1
print("PROBLEMS", problems)
