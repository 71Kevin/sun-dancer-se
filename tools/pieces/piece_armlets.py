"""Sun Dancer Armlets (slot 58): gold upper-arm cuffs with a raised rim along both edges and a small sun on the
outer side. The cuff follows the skin's own weights (rings with averaged weights let the triceps push through when
the arms swing); the rims are tubes along the cuff's real edges with the cuff's own weights and slider data, so they
hug it on every body (round tori stood off the oval arm as spikes on large presets), swept as closed loops (no
crease where they close)."""
import os
import pickle

import numpy as np

import mk_blend
import mk_geom
from ev_shapes import band, edge_tubes, rigid_bind, sun_disc
from mk_parts import Part

OFF = 0.18     # the inner face (OFF - THICK) stays outside the skin
THICK = 0.14
RIM_R = 0.1


def edge_rims(v, tt, w, D, name, a, ax):
    """Tubes along the two edges of the band (see ev_shapes.edge_tubes: evenly resampled, smoothed, closed sweeps),
    centred in its thickness; every ring takes its edge vertex's weights and slider data."""
    out = []
    for k, (rv, rt, ruv, idx) in enumerate(edge_tubes(v, tt, a, ax, THICK, RIM_R)):
        out.append(Part("%s%d" % (name, k), "ArmletsGold", rv, rt, ruv, "patch:gold", "gold", w[idx], D[idx]))
    return out


def build(body, proxy, log=print, work=None):
    work = work or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                "cache")
    skel = pickle.load(open(os.path.join(work, "ref", "skeleton.pkl"), "rb"))
    parts = []
    for side in (-1, 1):
        s = "L" if side < 0 else "R"
        a = skel["NPC %s UpperArm [%sUar]" % (s, s)][:3, 3]
        b = skel["NPC %s Forearm [%sLar]" % (s, s)][:3, 3]
        L = float(np.linalg.norm(b - a))
        s0, s1 = 0.5 * L, 0.5 * L + 1.7
        p, (v, tt, w, D, attrs, fr) = band(proxy, "armlet" + s, "ArmletsGold", a, b, s0, s1, OFF, THICK,
                                           "armlet" + s, "gold", cols=120, step=0.08, radius=3.8)
        for q in p:
            q.mat = "gold"
            if q.name.endswith(("_in", "_rim")):
                q.island = "patch:goldworn"
        parts += p
        ax, u, wv = fr
        parts += edge_rims(v, tt, w, D, "armletrim" + s, a, ax)
        tree = mk_blend.bvh(v, tt)
        outdir = mk_geom.normalize(np.array([side * 1.0, 0.0, 0.0]) - ax * ax[0] * side)
        c = a + ax * 0.5 * (s0 + s1)
        loc, nrm, face, dist = mk_blend.raycast(tree, (c + outdir * 15)[None], -outdir[None], 20.0)
        n = nrm[0] if np.dot(nrm[0], outdir) > 0 else -nrm[0]
        sv, st, suv = sun_disc(loc[0] + n * 0.06, n, -ax, 0.55, 0.16, rays=10, ray_len=0.25, rings=3)
        sw, sD = rigid_bind(sv, proxy, loc[0] - n * (OFF + THICK))
        parts.append(Part("armletsun" + s, "ArmletsGold", sv, st, suv, "patch:gold", "gold", sw, sD))
    log("armlets: %d parts" % len(parts))
    for p in parts:
        p.bones = list(proxy.bones)
        p.sliders = list(proxy.slider_names)
        p.shape = "ArmletsGold"
    return {"parts": parts}
