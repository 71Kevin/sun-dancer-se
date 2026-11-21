"""Sun Dancer Bracelets (slot 34): delicate gold sun bracelets at both wrists (they replace the forearm guards,
whose long plate bent and crumpled with the forearm's twist bones).

Per wrist: an engraved gold band with rolled edges and an eclipse sun on the outer side of the wrist (a small gold
sun round an obsidian disc in a thin ring), between two fine round bangles. Every piece of one wrist shares ONE set
of weights: the skin's own weights averaged round the wrist, kept to the forearm and its twist bones (no hand, no
upper arm). All those bones turn about the forearm's axis, so the set turns with the wrist as a rigid whole and
never bends, shears or crumples. The slider data follows the wrist's own shape (each ring keeps its distance from
the skin on every body). The band, the bangles and the rims are closed sweeps (no crease where a ring closes). The
same mesh is the first-person model."""
import os
import pickle

import numpy as np

import mk_blend
import mk_fit
import mk_geom
from ev_shapes import axial_rig, band, dome, edge_tubes, limb_grid, sun_disc, torus
from mk_parts import Part

OFF = 0.3        # the band's inner face (OFF - THICK) stays clear of the skin
THICK = 0.12
WIDTH = 0.8      # band width along the forearm
RIM_R = 0.075    # rolled edges, a little proud of the band
WIRE_R = 0.065   # the fine bangles
SHAPE = "BraceletsGold"


def arm_axis(skel, side):
    s = "L" if side < 0 else "R"
    return skel["NPC %s Forearm [%sLar]" % (s, s)][:3, 3], skel["NPC %s Hand [%sHnd]" % (s, s)][:3, 3]


def wrist_end(proxy, a, b):
    """Distance along the forearm axis where the body mesh ends at the wrist (the hand is another mesh)."""
    ax = mk_geom.normalize(b - a)
    best = None
    for loop in mk_geom.boundary_loops(proxy.t):
        c = proxy.v[loop].mean(0)
        if np.linalg.norm(c - b) < 6.0:
            best = float(np.dot(c - a, ax))
    return best if best is not None else float(np.linalg.norm(b - a)) - 4.5


def closed_tube(path, radius, seg, up):
    """A thin round tube along a closed loop (path without the repeated point)."""
    loop = np.vstack([path, path[:1]])
    v, t, uv, cols = mk_geom.sweep(loop, np.full(len(loop), radius), seg, up=tuple(up), closed=True)
    return v, t, uv


def bangle_loop(proxy, a, b, s, clearance, cols=96):
    """A loop round the wrist at distance s along the forearm, `clearance` off the skin, smoothed into an even oval;
    returns the loop and the skin points under it (for the slider data)."""
    p, n, t, ss, th, fr, rows, cc = limb_grid(proxy, a, b, s - 0.1, s + 0.1, 0.1, cols, radius=3.0)
    mid = rows // 2
    skin = p.reshape(rows, cc, 3)[mid]
    ring = skin + n.reshape(rows, cc, 3)[mid] * clearance
    for _ in range(12):     # an even oval: no little bumps from the skin's detail
        ring = 0.5 * ring + 0.25 * (np.roll(ring, 1, axis=0) + np.roll(ring, -1, axis=0))
    ring = mk_fit.push_out(ring, proxy, clearance * 0.9, iters=2)
    return ring, skin


def build(body, proxy, log=print, work=None):
    work = work or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                "cache")
    skel = pickle.load(open(os.path.join(work, "ref", "skeleton.pkl"), "rb"))
    parts = []
    for side in (-1, 1):
        tag = "L" if side < 0 else "R"
        a, b = arm_axis(skel, side)
        ax = mk_geom.normalize(b - a)
        s_wrist = wrist_end(proxy, a, b)
        s_c = s_wrist - 1.55                       # the band's centre: on the forearm, clear of the hand mesh
        rig = axial_rig(proxy, a, b, s_c - 2.5, s_c + 1.2, keep=("Forearm",), radius=3.2)
        W = rig((a + ax * s_c)[None])               # one set of weights for everything on this wrist

        def rigid(points, W=W):
            return np.repeat(W, len(points), 0)

        # the engraved band and its rolled edges
        # step 0.065: the band's cut lines fall between the grid's rings (a cut exactly on a ring of vertices, as
        # with 0.06, leaves a serrated edge)
        bp, (v, tt, w, D, attrs, fr) = band(proxy, "bracelet" + tag, SHAPE, a, b, s_c - WIDTH / 2, s_c + WIDTH / 2,
                                            OFF, THICK, "bracelet" + tag, "gold", cols=120, step=0.065, radius=3.0,
                                            weights_fn=rigid)
        for q in bp:
            q.mat = "gold"
            if q.name.endswith(("_in", "_rim")):
                q.island = "patch:goldworn"
        parts += bp
        for k, (rv, rt, ruv, idx) in enumerate(edge_tubes(v, tt, a, ax, THICK, RIM_R)):
            parts.append(Part("braceletrim%s%d" % (tag, k), SHAPE, rv, rt, ruv, "patch:gold", "gold", rigid(rv),
                              D[idx]))
        # two fine bangles either side of the band
        for k, ds in enumerate((-(WIDTH / 2 + 0.42), WIDTH / 2 + 0.36)):
            ring, skin = bangle_loop(proxy, a, b, s_c + ds, OFF + 0.06)
            gv, gt, guv = closed_tube(ring, WIRE_R, 8, ax)
            cols = 8 + 1
            idx = np.repeat(np.concatenate([np.arange(len(ring)), [0]]), cols)[:len(gv)]
            _, sD, _ = mk_fit.transfer(skin, proxy)
            gD = mk_fit.garment_conform(ring, sD, proxy)
            parts.append(Part("bangle%s%d" % (tag, k), SHAPE, gv, gt, guv, "patch:gold", "gold", rigid(gv), gD[idx]))
        # the eclipse sun on the outer side of the wrist
        _, u, wdir = fr
        outdir = mk_geom.normalize(np.array([side * 1.0, 0.0, 0.0]) - ax * ax[0] * side)
        c = a + ax * s_c
        tree = mk_blend.bvh(v, tt)
        loc, nrm, _, dist = mk_blend.raycast(tree, (c + outdir * 15)[None], -outdir[None], 20.0)
        n = nrm[0] if np.dot(nrm[0], outdir) > 0 else -nrm[0]
        base = loc[0] + n * 0.05
        sv, st, suv = sun_disc(base, n, -ax, 0.5, 0.14, rays=12, ray_len=0.22, rings=3)
        _, sD = mk_fit.transfer(loc[0][None] - n[None] * (OFF + 0.05), proxy)[:2]
        sD = mk_fit.garment_conform(base[None], sD, proxy)
        parts.append(Part("wristsun" + tag, SHAPE, sv, st, suv, "wristsun", "gold", rigid(sv),
                          np.repeat(sD, len(sv), 0)))
        ov, ot, ouv = dome(base + n * 0.1, n, 0.25, 0.09, seg=20, rings=4)
        parts.append(Part("wristsun_eclipse" + tag, SHAPE, ov, ot, ouv, "patch:obsidian", "gem", rigid(ov),
                          np.repeat(sD, len(ov), 0)))
        rv, rt, ruv = torus(base + n * 0.12, n, -ax, 0.29, 0.03, 32, 6)
        parts.append(Part("wristsun_ring" + tag, SHAPE, rv, rt, ruv, "patch:gold", "gold", rigid(rv),
                          np.repeat(sD, len(rv), 0)))
    log("bracelets: %d parts" % len(parts))
    for p in parts:
        p.bones = list(proxy.bones)
        p.sliders = list(proxy.slider_names)
        p.shape = SHAPE
    return {"parts": parts}
