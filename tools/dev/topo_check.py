"""Open boundaries and bone regions of the UBE body (development helper, plain Python)."""
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import mk_geom  # noqa: E402

ref = pickle.load(open(sys.argv[1], "rb"))
sh = ref["shapes"]["BaseShape"]
v, t = sh["v"], sh["t"]
q = np.round(v / 1e-4).astype(np.int64)
_, first, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
inv = inv.ravel()
wv, wt = v[first], inv[t]
good = (wt[:, 0] != wt[:, 1]) & (wt[:, 1] != wt[:, 2]) & (wt[:, 2] != wt[:, 0])
wt = wt[good]
print("welded verts", len(wv), "tris", len(wt))
for loop in mk_geom.boundary_loops(wt):
    c = wv[loop].mean(0)
    print("loop %4d verts center (%6.2f %6.2f %6.2f) radius %.2f" % (len(loop), *c, np.linalg.norm(wv[loop] - c, axis=1).mean()))
W = np.zeros((len(wv), len(sh["bones"])))
np.add.at(W, inv, sh["w"])
for j, b in enumerate(sh["bones"]):
    m = W[:, j] > 0.3
    if m.sum() and any(k in b for k in ("Pussy", "Vagina", "Clit", "Anus", "Anal")):
        p = wv[m]
        print("%-22s n=%5d x[%6.2f %6.2f] y[%6.2f %6.2f] z[%6.2f %6.2f]" % (b, m.sum(), p[:, 0].min(), p[:, 0].max(),
                                                                       p[:, 1].min(), p[:, 1].max(), p[:, 2].min(),
                                                                       p[:, 2].max()))
print("partitions", sh["partitions"], "tri counts", np.bincount(np.asarray(sh["partition_tris"])) if len(sh["partition_tris"]) else None)
pt = np.asarray(sh["partition_tris"])
if len(pt):
    sel = t[pt == 1]
    p = v[np.unique(sel)]
    print("partition 1 bbox", p.min(0).round(2), p.max(0).round(2), len(sel), "tris")
