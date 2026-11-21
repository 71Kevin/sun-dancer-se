"""Face orientation of every shape in the built ShapeData NIFs: share of triangles facing away from the body
(outer garment layers should be ~1, linings ~0). Runs inside Blender."""
import os
import pickle
import sys

import numpy as np

TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(TOOLS, "lib"))
import mk_blend  # noqa: E402
import mk_fit  # noqa: E402
import mk_geom  # noqa: E402
import mk_nif  # noqa: E402

work, sd = sys.argv[sys.argv.index("--") + 1:][:2]
body = mk_fit.Body(pickle.load(open(os.path.join(work, "ref", "body.pkl"), "rb")))
feet = mk_fit.Body(pickle.load(open(os.path.join(work, "ref", "feet.pkl"), "rb")), shape_names=("Feet",))
v = np.vstack([body.v, feet.v])
t = np.vstack([body.t, feet.t + len(body.v)])
tree = mk_blend.bvh(v, t)
for stem in ("Wrap", "Loincloth", "LoinclothNoBriefs", "Briefs", "Sandals", "Bracelets", "Armlets", "Veils",
             "Collar", "Navel"):
    nif = mk_nif.NifFile(os.path.join(sd, stem + ".nif"))
    for s in nif.shapes:
        sh = mk_nif.read_shape(s)
        if sh["name"] in ("BaseShape", "Feet"):
            continue
        fv = sh["v"][sh["t"]].mean(1)
        fn = mk_geom.face_normals(sh["v"], sh["t"])
        loc, _, _, _ = mk_blend.nearest(tree, fv)
        away = ((fv - loc) * fn).sum(1) > 0
        print("%-10s %-16s tris %6d  facing away from body %.2f" % (stem, sh["name"], len(fv), away.mean()))
