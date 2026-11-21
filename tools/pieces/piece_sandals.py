"""Sun Dancer Sandals (slot 37): flat leather sandals on the UBE feet (the feet are part of the mesh) with a
toe strap, crossed instep straps, an ankle ring with a gold stud, a heel strap, and straps crossing up the calf to a
gold-studded calf band. A small high-heels offset lifts the character by the sole thickness (RaceMenu)."""
import os
import pickle

import numpy as np

import mk_blend
import mk_fit
import mk_geom
from mk_parts import Part
from piece_wrap import dome, ribbon, sun_disc

STRAP_W = 0.52
STRAP_T = 0.1
STRAP_OFF = 0.12
SOLE_T = 0.32
HH = 0.3
SEAM = (10.6, 11.4)          # feet mesh below, body mesh above (z)
ANKLE_Z = 9.3
CALF_TOP = 25.0
TURNS = 1.25


class Union:
    """Weights and slider data in the union of the body's and the feet's bones and sliders."""

    def __init__(self, proxy, feet):
        self.bones = list(proxy.bones) + [b for b in feet.bones if b not in proxy.bones]
        self.sliders = list(proxy.slider_names) + [s for s in feet.slider_names if s not in proxy.slider_names]
        self.pb = [self.bones.index(b) for b in proxy.bones]
        self.fb = [self.bones.index(b) for b in feet.bones]
        self.ps = [self.sliders.index(s) for s in proxy.slider_names]
        self.fs = [self.sliders.index(s) for s in feet.slider_names]
        self.proxy, self.feet = proxy, feet

    def bind(self, v, t=None, smooth=2):
        n = len(v)
        W = np.zeros((n, len(self.bones)))
        D = np.zeros((n, len(self.sliders), 3), dtype=np.float32)
        k = np.clip((v[:, 2] - SEAM[0]) / (SEAM[1] - SEAM[0]), 0, 1)
        k = k * k * (3 - 2 * k)
        wp, Dp, _ = mk_fit.transfer(v, self.proxy, t, smooth if t is not None else 0)
        wf, Df, _ = mk_fit.transfer(v, self.feet, t, smooth if t is not None else 0)
        W[:, self.pb] += wp * k[:, None]
        W[:, self.fb] += wf * (1 - k)[:, None]
        D[:, self.ps] += Dp * k[:, None, None]
        D[:, self.fs] += Df * (1 - k)[:, None, None]
        W /= np.maximum(W.sum(1, keepdims=True), 1e-9)
        return W, D


def combined_tree(proxy, feet):
    sel = proxy.t[(proxy.v[proxy.t][:, :, 2].min(1) > SEAM[0] - 0.4)]
    v = np.vstack([proxy.v, feet.v])
    t = np.vstack([sel, feet.t + len(proxy.v)])
    return mk_blend.bvh(v, t)


def on_surface(tree, pts, off, iters=3, smooth=True):
    p = np.asarray(pts, dtype=np.float64)
    for _ in range(iters):
        loc, nrm, _, _ = mk_blend.nearest(tree, p)
        p = loc + nrm * off
        if smooth:
            for _ in range(4):
                p[1:-1] += 0.5 * (0.5 * (p[:-2] + p[2:]) - p[1:-1])
    loc, nrm, _, _ = mk_blend.nearest(tree, p)
    return loc + nrm * off, nrm


def fan(tree, origin, a, b, angles, reach=10.0):
    """Points where rays from origin, in the plane of unit vectors a and b, leave the surface."""
    d = np.outer(np.cos(angles), a) + np.outer(np.sin(angles), b)
    o = np.repeat(np.asarray(origin, dtype=np.float64)[None], len(d), 0)
    loc, nrm, _, dist = mk_blend.raycast(tree, o, d, reach)
    ok = np.isfinite(dist)
    return loc[ok], nrm[ok]


def strap(name, tree, pts, union, closed=False, width=STRAP_W, mat="leather", island="patch:strap"):
    if closed:
        pts = np.vstack([pts, pts[:1]])
    path = mk_geom.resample_polyline(pts, max(8, int(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum() / 0.25)))
    path, nrm = on_surface(tree, path, STRAP_OFF + STRAP_T * 0.5)
    v, t, uv = ribbon(path, nrm, width, STRAP_T, segs=8, caps=not closed)
    w, D = union.bind(v, t)
    # straps lie over the instep and ankle: no toe weights (next to the toes they would swing with a bent toe)
    toe = np.array(["Toe" in b for b in union.bones])
    if toe.any():
        w[:, toe] = 0
        empty = w.sum(1) <= 1e-6
        for side, tag in ((-1, "NPC L Foot [Lft ]"), (1, "NPC R Foot [Rft ]")):
            if tag in union.bones:
                w[empty & (np.sign(v[:, 0]) == side), union.bones.index(tag)] = 1.0
        w /= np.maximum(w.sum(1, keepdims=True), 1e-9)
    return Part(name, "Sandals", v, t, uv, island, mat, w, D)


def convex_hull_2d(p):
    """Andrew's monotone chain; returns the hull counter-clockwise."""
    p = np.unique(np.round(np.asarray(p, dtype=np.float64), 5), axis=0)
    p = p[np.lexsort((p[:, 1], p[:, 0]))]

    def half(pts):
        h = []
        for q in pts:
            while len(h) >= 2 and np.cross(h[-1] - h[-2], q - h[-2]) <= 0:
                h.pop()
            h.append(q)
        return h
    lower = half(p)
    upper = half(p[::-1])
    return np.array(lower[:-1] + upper[:-1])


def foot_info(feet, side):
    m = np.sign(feet.v[:, 0]) == side
    fv = feet.v[m]
    return fv, fv[:, 0].mean()


def sole_parts(feet, union, side):
    fv, xc = foot_info(feet, side)
    bottom = fv[fv[:, 2] < fv[:, 2].min() + 0.9]
    hull = convex_hull_2d(bottom[:, :2])
    c = hull.mean(0)
    hull = hull + mk_geom.normalize(hull - c) * 0.32
    ring = mk_geom.resample_polyline(mk_geom.catmull_rom(hull, 4, closed=True), 72, closed=True)
    z_top = fv[:, 2].min() + 0.02
    z_bot = z_top - SOLE_T
    n = len(ring)
    top = np.c_[ring, np.full(n, z_top)]
    bot = np.c_[ring, np.full(n, z_bot)]
    ct, cb = np.r_[c, z_top], np.r_[c, z_bot]
    v = np.vstack([top, bot, ct, cb])
    tris = []
    for i in range(n):
        j = (i + 1) % n
        tris += [[2 * n, i, j], [2 * n + 1, n + j, n + i], [i, n + i, n + j], [i, n + j, j]]
    t = np.array(tris)
    if side > 0:
        t = t[:, ::-1].copy()
    nrm = mk_geom.vertex_normals(v, t)
    if np.dot(nrm[2 * n], [0, 0, 1]) < 0:
        t = t[:, ::-1].copy()
    uv = np.vstack([ring, ring, c[None], c[None]])
    # each sole vertex takes the weights of the skin straight above it, so the sole bends with the ball of the
    # foot and the toes exactly like the foot does (nearest-point weights put toe weights under the ball)
    up = np.tile([0, 0, 1.0], (len(v), 1))
    o = np.c_[v[:, :2], np.full(len(v), z_bot - 0.5)]
    loc, _, _, dist = mk_blend.raycast(feet.tree, o, up, 6.0)
    miss = ~np.isfinite(dist)
    if miss.any():
        l2, _, _, _ = mk_blend.nearest(feet.tree, np.c_[v[miss, :2], np.full(miss.sum(), z_top + 0.05)])
        loc[miss] = l2
    w, D = union.bind(loc)
    return Part("sole%d" % side, "Sandals", v, t, uv, "patch:sole", "sole", w, D), (ring, z_top, xc, fv)


def build(body, proxy, log=print, work=None):
    work = work or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                "cache")
    fref = pickle.load(open(os.path.join(work, "ref", "feet.pkl"), "rb"))
    skel = pickle.load(open(os.path.join(work, "ref", "skeleton.pkl"), "rb"))
    feet = mk_fit.Body(fref, shape_names=("Feet",))
    union = Union(proxy, feet)
    tree = combined_tree(proxy, feet)
    parts = []
    for side in (-1, 1):
        sole, (ring, z_top, xc, fv) = sole_parts(feet, union, side)
        parts.append(sole)
        y0, y1 = fv[:, 1].min(), fv[:, 1].max()
        L = y1 - y0
        up = np.array([0, 0, 1.0])
        out = np.array([side * 1.0, 0, 0])
        # toe strap: an arch over the metatarsals, 1.6 units behind the toe joint (so the strap and the skin under
        # it follow the foot bone only and a bending toe does not drag it)
        toe_j = skel["NPC %s Toe0 [%sToe]" % ("L" if side < 0 else "R", "L" if side < 0 else "R")][:3, 3]
        yt = min(y1 - 0.24 * L, toe_j[1] - 1.6)
        sec = fv[np.abs(fv[:, 1] - yt) < 0.4]
        o = np.array([sec[:, 0].mean(), yt, z_top + 1.2])
        p, n = fan(tree, o, up, out, np.linspace(-1.75, 1.75, 40))
        parts.append(strap("toe%d" % side, tree, p, union))
        # crossed instep straps
        ym = y1 - 0.5 * L
        sec = fv[np.abs(fv[:, 1] - ym) < 0.5]
        top_z = sec[:, 2].max()
        xin, xout = sec[:, 0].min(), sec[:, 0].max()
        if side < 0:
            xin, xout = xout, xin
        for k, (ya, yb) in enumerate(((ym - 1.6, ym + 1.4), (ym + 1.4, ym - 1.6))):
            a = np.array([xin, ya, z_top + 0.35])
            b = np.array([xout, yb, z_top + 0.35])
            mid = np.array([sec[:, 0].mean(), 0.5 * (ya + yb), top_z + 0.4])
            ctrl = mk_geom.catmull_rom(np.array([a, 0.5 * (a + mid) + np.array([0, 0, 0.8]), mid,
                                                 0.5 * (b + mid) + np.array([0, 0, 0.8]), b]), 6)
            parts.append(strap("instep%d_%d" % (side, k), tree, ctrl, union))
        # ankle ring
        ank = fv[np.abs(fv[:, 2] - ANKLE_Z) < 0.3]
        ac = np.array([ank[:, 0].mean(), ank[:, 1].mean(), ANKLE_Z])
        angles = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        p, n = fan(tree, ac, np.array([0, 1.0, 0]), out, angles)
        parts.append(strap("ankle%d" % side, tree, p, union, closed=True))
        # dancer's anklet: small sun coins hanging from the ring at the sides (the stud sits in front, and the instep
        # right under the front of the ring would catch hanging coins)
        from ev_shapes import coin as coin_shape
        from piece_wrap import torus as torus_shape
        for k, a in enumerate((-1.25, -0.7, 0.7, 1.25, 1.8)):
            i = int(round((a % (2 * np.pi)) / (2 * np.pi) * len(angles))) % len(angles)
            nn = mk_geom.normalize(n[i] - np.array([0, 0, 1.0]) * n[i][2])
            top = p[i] + nn * (STRAP_OFF + STRAP_T + 0.06) - np.array([0, 0, 1.0]) * (STRAP_W * 0.5)
            jv, jt, juv = torus_shape(top, np.cross(nn, [0, 0, 1.0]), [0, 0, 1.0], 0.08, 0.022, 10, 4)
            cv, ct, cuv = coin_shape(top - np.array([0, 0, 0.36]) + nn * 0.04, nn, [0, 0, 1.0], radius=0.26,
                                     thick=0.045)
            w, D = union.bind(np.repeat(p[i][None], 1, 0))
            for nm, vv, tt, uu, isl in (("anklering", jv, jt, juv, "patch:gold"), ("anklecoin", cv, ct, cuv, "coin")):
                parts.append(Part("%s%d_%d" % (nm, side, k), "SandalsGold", vv, tt, uu, isl, "gold",
                                  np.repeat(w, len(vv), 0), np.repeat(D, len(vv), 0)))
        # heel strap: from the back of the sole up to the ankle ring
        heel = fv[fv[:, 1] < y0 + 0.6]
        hb = np.array([heel[:, 0].mean(), y0 - 0.2, z_top + 0.4])
        ht = np.array([ac[0], ank[:, 1].min() - 0.2, ANKLE_Z])
        parts.append(strap("heel%d" % side, tree, mk_geom.catmull_rom(np.array([hb, 0.5 * (hb + ht) +
                                                                               np.array([0, -0.6, 0]), ht]), 8),
                           union))
        # calf: two helices of opposite hand from the ankle ring to the calf band
        knee = skel["NPC %s Calf [%sClf]" % ("L" if side < 0 else "R", "L" if side < 0 else "R")][:3, 3]
        foot_j = skel["NPC %s Foot [%sft ]" % ("L" if side < 0 else "R", "L" if side < 0 else "R")][:3, 3]
        ax = mk_geom.normalize(knee - foot_j)
        for hand in (1, -1):
            zz = np.linspace(ANKLE_Z, CALF_TOP, 60)
            s = (zz - foot_j[2]) / ax[2]
            centers = foot_j[None] + np.outer(s, ax)
            ang = np.pi * 0.5 + hand * np.linspace(0, 2 * np.pi * TURNS, len(zz))
            dirs = np.stack([np.cos(ang) * side, np.sin(ang), np.zeros_like(ang)], 1)
            loc, nrm, _, dist = mk_blend.raycast(tree, centers, dirs, 12.0)
            ok = np.isfinite(dist)
            parts.append(strap("calf%d_%d" % (side, hand), tree, loc[ok], union))
        cc = foot_j + ax * ((CALF_TOP + 0.4 - foot_j[2]) / ax[2])
        a2 = mk_geom.normalize(np.cross(ax, out))
        p, n = fan(tree, cc, a2, np.cross(ax, a2), np.linspace(0, 2 * np.pi, 56, endpoint=False))
        parts.append(strap("calfband%d" % side, tree, p, union, closed=True, width=0.75))
        # gold studs: ankle front and calf band outer side
        for name, center, d0 in (("anklestud", ac, np.array([0, 1.0, 0])), ("calfstud", cc, out)):
            loc, nrm, _, dist = mk_blend.raycast(tree, (center)[None], d0[None], 10.0)
            if not np.isfinite(dist[0]):
                continue
            c = loc[0] + nrm[0] * (STRAP_OFF + STRAP_T + 0.05)
            if name == "calfstud":
                sv, st, suv = sun_disc(c, nrm[0], ax, 0.42, 0.14, rays=10, ray_len=0.2, rings=3)
            else:
                sv, st, suv = dome(c, nrm[0], 0.34, 0.16)
            w, D = union.bind(np.repeat(loc[0][None], len(sv), 0))
            parts.append(Part("%s%d" % (name, side), "SandalsGold", sv, st, suv, "patch:gold", "gold", w, D))
    log("sandals: %d parts" % len(parts))
    for p in parts:
        p.bones = list(union.bones)
        p.sliders = list(union.sliders)
        p.shape = "SandalsGold" if p.mat in ("gold", "gem") else "Sandals"
    skin = feet_skin(fref, feet)
    return {"parts": parts, "bones": union.bones, "ref_body": feet, "skin": skin,
            "extras": {"Sandals": [("SDTA", '[{"name":"NPC","pos":[0,0,%.2f]}]' % HH)]}}


def feet_skin(fref, feet):
    sh = fref["shapes"]["Feet"]
    sliders, cols = [], []
    for attrs, data in fref["sliders"]:
        for dname, tgt, _ in data:
            if tgt != "Feet":
                continue
            idx, d = fref["diffs"][(attrs["name"], tgt)]
            dense = np.zeros((len(sh["v"]), 3), dtype=np.float32)
            if len(idx):
                dense[idx] = d
            sliders.append(attrs["name"])
            cols.append(dense)
    n = mk_geom.vertex_normals(feet.v, feet.t)[feet.inv[np.arange(len(sh["v"]))]]
    data = {"name": "Feet", "v": sh["v"], "t": sh["t"], "uv": sh["uv"], "n": n, "bones": sh["bones"], "w": sh["w"],
            "sliders": sliders, "D": np.stack(cols, 1)}
    attrs = [a for a, _ in fref["sliders"]]
    return data, sh, attrs
