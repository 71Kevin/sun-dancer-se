"""Sun Dancer Briefs (slot 52): a low linen tanga with a gold-embroidered trim and a gold ring at each hip, worn on
its own. The same briefs are part of the Loincloth with Briefs (piece_loincloth.build_with_briefs); the third choice
is the Loincloth without them. All three use slot 52, so one replaces the other.

The shape (round 5; the round-4 briefs read as an odd shape in game: a shallow, wavy-topped front sitting on the pubis,
the crotch spread over the inner thighs, an uneven trim):
- the waistband follows the line of the loincloth's belt: low in front (WAIST, ten units under the navel) and rising
  over the hips, so it stays under the belt when the loincloth is worn over it;
- the legs are cut high along the groin creases down to a narrow gusset (about 2.6 wide under the crotch, GUSSET, on
  the part of the surface that faces down), and the back is a cheeky cut across the buttocks (LEG_FRONT, LEG_BACK:
  the lines as drawn on the front and back views, flowing into the gusset without a corner);
- the sides are narrow bands ending at a gold ring on each hip: the front band and the back band end at its wire, the
  band pinched to the ring's opening, the skin showing inside the ring;
- the cloth is shaped on a smooth envelope of the pelvis (vulva closed, cleft bridged), draped, stretched over its
  hollows away from the edges, then cut along those smooth curves (clean edges, no ripple);
- an even trim strip (TRIM_W) runs along every edge in texture islands of its own, laid out by arc length and distance
  from the edge (make_textures embroiders it like the Wrap's cups); the rest is plain linen.

Worn over the Wrap's whole lower body: every vertex takes the weights of the skin under it, blended across the midline
(the bridged cleft) and kept to 4 bones that fade out where one has to go (mk_fit.limit_influences); it stands OFF off
the envelope, a little more over the vulva, and stays in front of the body under every slider."""
import numpy as np

import mk_blend
import mk_export
import mk_fit
import mk_geom
import piece_wrap as W
from mk_parts import Part, bind, solid_parts

OFF = 0.17
THICK = 0.1
TRIM_W = 0.46
Y_SPLIT = -3.5          # front and back texture islands; the axis the waist angle is measured round
RING_R = 0.55           # hip rings: radius of the wire's centre line (the wire is 0.11 thick)
RING_WIRE = 0.11
# waistband height by angle round the pelvis (0 = front, pi = back)
WAIST = [(0.0, 70.6), (0.4, 70.9), (0.8, 71.6), (1.2, 72.4), (1.57, 72.95), (2.0, 73.1), (2.6, 73.15), (np.pi, 73.2)]
# leg lines as drawn on the front and on the back view: height by |x|, z0 + c (|x| - x0)^p through (x1, z1), the
# side (under the ring) at x1. In front a high cut along the groin creases, steep into the crotch; behind a cheeky
# cut across the buttocks, flatter in the middle. The crotch itself is the gusset strip.
LEG_FRONT = (1.3, 64.4, 12.8, 72.0, 0.72)
LEG_BACK = (2.0, 65.6, 12.8, 72.0, 1.25)
# gusset: half width by y under the crotch (narrowest just behind the vulva, opening into the front and back)
GUSSET = [(-9.0, 2.2), (-7.0, 1.7), (-5.5, 1.48), (-4.0, 1.36), (-2.5, 1.3), (-1.0, 1.27), (1.0, 1.32), (2.5, 1.6),
          (4.0, 2.3), (6.0, 3.4)]
INTIMATE = W.INTIMATE
_CACHE = {}


# ---------------------------------------------------------------- outline
def smin(a, b, k):
    h = np.clip(k - np.abs(a - b), 0, None) / k
    return np.minimum(a, b) - h * h * k * 0.25


def smax(a, b, k):
    return -smin(-a, -b, k)


def smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def open_curve(ctrl, samples=24):
    dense = mk_geom.catmull_rom(np.array(sorted(ctrl), dtype=np.float64), samples)
    order = np.argsort(dense[:, 0])
    a, g = dense[order, 0], dense[order, 1]

    def f(x):
        return np.interp(np.asarray(x), a, g)
    return f


waist = mk_fit.sym_curve(WAIST)
gusset = open_curve(GUSSET)


def leg_curve(spec, dip=0.0):
    """z0 + c (|x| - x0)^p beyond x0; inside x0 the line keeps dipping (dip per unit squared) so that it runs into
    the gusset's sides instead of meeting them square."""
    x0, z0, x1, z1, p = spec
    c = (z1 - z0) / (x1 - x0) ** p

    def f(ax):
        u = np.asarray(ax) - x0
        return z0 + c * np.clip(u, 0, None) ** p - dip * np.clip(-u, 0, None) ** 2
    return f


leg_front = leg_curve(LEG_FRONT)
leg_back = leg_curve(LEG_BACK, dip=1.6)


def leg_height(v):
    """Height of the leg line over each point: the front curve on the front, the back one behind, blended over the
    sides (where both reach the ring's band)."""
    th = np.arctan2(v[:, 0], v[:, 1] - Y_SPLIT)
    a = smoothstep((np.abs(th) - (np.pi / 2 - 0.35)) / 0.7)
    ax = np.abs(v[:, 0])
    return (1 - a) * leg_front(ax) + a * leg_back(ax)


def ring_frames(env):
    """Centre (on the cloth), normal, up and the direction along the band (toward the back) of each hip ring."""
    out = []
    for side in (-1, 1):
        o = np.array([0.0, Y_SPLIT, 72.0])
        d = np.array([side * 1.0, 0.0, 0.0])
        loc, _, _, _ = mk_blend.raycast(env.tree, (o + d * 30)[None], -d[None], 40.0)
        z = 0.5 * (float(waist(np.array([np.pi / 2]))[0]) + float(leg_height(loc)[0]))
        o = np.array([0.0, Y_SPLIT, z])
        d = np.array([side * 1.0, 0.0, 0.0])
        loc, nrm, _, _ = mk_blend.raycast(env.tree, (o + d * 30)[None], -d[None], 40.0)
        n = nrm[0] if np.dot(nrm[0], d) > 0 else -nrm[0]
        up = mk_geom.normalize(np.array([0, 0, 1.0]) - n * n[2])
        along = mk_geom.normalize(side * np.cross(n, up))
        out.append((loc[0] + n * (OFF + 0.04), n, up, along, loc[0]))
    return out


def soft_normals(v, t, iters=10):
    """Vertex normals relaxed over a few rings: the gusset's edge follows them, and raw normals made it wiggle."""
    n = mk_geom.vertex_normals(v, t)
    return mk_geom.normalize(mk_geom.smooth(n, mk_geom.neighbor_pairs(t, len(v)), iters, 0.5))


def outline(v, rings, n):
    """Signed distance-like field of the briefs' outline (>= 0 inside) on a surface with vertex normals n."""
    th = np.arctan2(v[:, 0], v[:, 1] - Y_SPLIT)
    f_waist = waist(th) - v[:, 2]
    f_leg = v[:, 2] - leg_height(v)
    # the gusset: a strip under the crotch, on the surface that faces down (the inner thighs' faces stand about as far
    # from the midline as the strip is wide: a cut by height ran down them in slivers, or nipped the grooves beside
    # the perineum, where the crotch is lower than in the middle)
    f_gus = smin(gusset(v[:, 1]) - np.abs(v[:, 0]), 2.0 * (-n[:, 2] - 0.35), 0.6)
    f = smin(f_waist, smax(f_leg, f_gus, 2.4), 0.8)
    for c, n, up, along, _ in rings:
        d = v - c
        s, h, o = d @ along, d @ up, d @ n
        near = (np.abs(s) < 2.0) & (np.abs(o) < 3.0) & (np.abs(h) < 3.4)
        k = smoothstep((np.abs(s) - RING_R) / 0.9)
        # the band cut through inside the ring's wire, and pinched toward it (to 0.84 at the wire)
        f_ring = np.minimum(np.hypot(s, h) - RING_R, (0.42 + 3.0 * k) - np.abs(h))
        f = np.where(near, np.minimum(f, f_ring), f)
    return f


# ---------------------------------------------------------------- surface
def biggest(v, t):
    e = mk_geom.unique_edges(t)
    lab = mk_geom.components(len(v), e[:, 0], e[:, 1])
    big = np.bincount(lab[t[:, 0]]).argmax()
    v, t, _, _ = mk_geom.compact(v, t[lab[t[:, 0]] == big])
    return v, t


def away_from_zero(f, eps=0.03):
    """Keeps the cut from passing right by a vertex (sliver triangles)."""
    return np.where(np.abs(f) < eps, np.where(f >= 0, eps, -eps), f)


def cloth_surface(env, rings):
    """The cloth over the briefs' region (with a margin round it): draped off the envelope, smoothed, stretched over the
    hollows (the cleft fully, the rest gently, never at the edges where the elastic holds it to the skin)."""
    v, t = env.v, env.t
    keep = (v[t][:, :, 2].max(1) < 76.5) & (v[t][:, :, 2].min(1) > 60.0)
    v, t, _, _ = mk_geom.compact(v, t[keep])
    f = outline(v, rings, soft_normals(v, t))
    v, t, _, _ = mk_geom.compact(v, t[(f[t] > -2.5).any(1)])
    v, t = biggest(v, t)
    rim = np.zeros(len(v), dtype=bool)
    rim[np.unique(mk_geom.boundary_edges(t))] = True
    pairs = mk_geom.neighbor_pairs(t, len(v))
    v = mk_fit.drape(v, t, env, OFF, iters=60, lam=0.45, fixed=rim)
    v = mk_geom.taubin(v, pairs, iters=12, fixed=rim)
    k_in = smoothstep(outline(v, rings, soft_normals(v, t)) / 1.5)
    cm = cleft_mask(v)
    k = np.maximum(cm, 0.45 * k_in) * np.maximum(k_in, cm)
    v = mk_fit.fill_concave(v, t, k, iters=220)
    v = mk_fit.push_out(v, env, OFF, iters=2)
    return v, t


def cut_outline(v, t, rings):
    tree = mk_blend.bvh(v, t)
    f = away_from_zero(outline(v, rings, soft_normals(v, t)))
    v2, t2, _ = mk_geom.iso_cut(v, t, f)
    v2, t2 = biggest(v2, t2)
    for _ in range(3):
        v2 = mk_geom.smooth_boundary(v2, t2, iters=6, lam=0.5)
        loc, _, _, _ = mk_blend.nearest(tree, v2)
        v2 = loc
    return v2, t2


def edge_distance(v, t, step=0.02):
    """Distance of every vertex to the edge, measured to the boundary curves themselves (the graph distance along mesh
    edges zigzags, and the trim's inner line rippled with it)."""
    pts = []
    for loop in mk_geom.boundary_loops(t):
        p = v[np.append(loop, loop[0])]
        L = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
        s = np.arange(0, L[-1], step)
        pts.append(np.stack([np.interp(s, L, p[:, j]) for j in range(3)], 1))
    pts = np.vstack(pts)
    d = mk_geom.edge_distance(v, t)
    near = np.nonzero(d < 3 * TRIM_W)[0]
    j = nearest_sample(v[near], pts)
    d[near] = np.minimum(d[near], np.linalg.norm(v[near] - pts[j], axis=1))
    return d


def split_trim(v, t):
    """The trim strip (within TRIM_W of the edge) and the rest, cut apart along a smooth line but kept as one welded
    mesh. Returns v, t, the distance from the edge of every vertex and the trim flag of every triangle."""
    d = edge_distance(v, t)
    f = away_from_zero(TRIM_W - d)
    va, ta, aa = mk_geom.iso_cut(v, t, f, {"d": d})
    vb, tb, ab = mk_geom.iso_cut(v, t, -f, {"d": d})
    vv = np.vstack([va, vb])
    tt = np.vstack([ta, tb + len(va)])
    dd = np.concatenate([aa["d"], ab["d"]])
    flag = np.concatenate([np.ones(len(ta), dtype=bool), np.zeros(len(tb), dtype=bool)])
    # weld the two halves along the cut (the same interpolation made both copies of each cut vertex)
    q = np.round(vv / 1e-5).astype(np.int64)
    _, first, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    t2 = inv[tt]
    ok = (t2[:, 0] != t2[:, 1]) & (t2[:, 1] != t2[:, 2]) & (t2[:, 2] != t2[:, 0])
    return vv[first], t2[ok], dd[first], flag[ok]


# ---------------------------------------------------------------- texture islands
def boundary_segments(v, t, rings):
    """The boundary loops split where the bands end at the rings (under the wire): the trim's islands."""
    loops = mk_geom.boundary_loops(t)
    ends = []
    for c, n, up, along, _ in rings:
        for sg in (-1, 1):
            ends.append(c + along * sg * RING_R)
    segs = []
    for loop in loops:
        p = v[loop]
        cut = sorted({int(np.argmin(np.linalg.norm(p - e, axis=1))) for e in ends
                      if np.linalg.norm(p - e, axis=1).min() < 0.8})
        if not cut:
            cut = [int(np.argmin(p[:, 1]))]
        for a, b in zip(cut, cut[1:] + [cut[0] + len(loop)]):
            idx = loop[np.arange(a, b + 1) % len(loop)]
            segs.append(idx)
    return segs


def sample_segments(v, segs, step=0.02):
    pts, sid, arc = [], [], []
    for k, idx in enumerate(segs):
        p = v[idx]
        L = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
        s = np.arange(0, L[-1] + 1e-9, step)
        pts.append(np.stack([np.interp(s, L, p[:, j]) for j in range(3)], 1))
        sid.append(np.full(len(s), k))
        arc.append(s)
    return np.vstack(pts), np.concatenate(sid), np.concatenate(arc)


def nearest_sample(q, pts, chunk=256):
    out = np.empty(len(q), dtype=np.int64)
    for a in range(0, len(q), chunk):
        d = ((q[a:a + chunk, None, :] - pts[None]) ** 2).sum(2)
        out[a:a + chunk] = np.argmin(d, 1)
    return out


def island_layout(v, t, d, trim, rings):
    """Island name of every triangle and a uv function per island: the trim by (arc length, distance from the edge),
    the front and back by an unwrap."""
    segs = boundary_segments(v, t, rings)
    pts, sid, arc = sample_segments(v, segs)
    labels = np.empty(len(t), dtype=object)
    cen = v[t].mean(1)
    tri_seg = sid[nearest_sample(cen[trim], pts)]
    labels[np.nonzero(trim)[0]] = ["brieftrim%d" % s for s in tri_seg]
    body = ~trim
    labels[body] = np.where(cen[body, 1] > Y_SPLIT, "briefs_0", "briefs_1")
    uv_of = {}
    for k in range(len(segs)):
        sel_pts = sid == k

        def f(ids, sel_pts=sel_pts):
            j = nearest_sample(v[ids], pts[sel_pts])
            return np.stack([arc[sel_pts][j], d[ids]], 1)
        uv_of["brieftrim%d" % k] = f
    method = unwrap_method()
    for name in ("briefs_0", "briefs_1"):
        def g(ids, name=name):
            sel = labels == name
            tt = t[sel]
            used = np.unique(tt)
            remap = -np.ones(len(v), dtype=np.int64)
            remap[used] = np.arange(len(used))
            uv = mk_blend.unwrap(v[used], remap[tt], method=method)
            return uv[remap[ids]]
        uv_of[name] = g
    return labels, uv_of


def unwrap_method():
    """Blender's minimum-stretch unwrap (4.3+) keeps areas as well as angles; the conformal one otherwise."""
    try:
        import bpy
        items = bpy.ops.uv.unwrap.get_rna_type().properties["method"].enum_items.keys()
        return "MINIMUM_STRETCH" if "MINIMUM_STRETCH" in items else "CONFORMAL"
    except Exception:
        return "CONFORMAL"


def island_splitter(labels, uv_of):
    def split(outer):
        out = []
        for name in sorted(set(labels)):
            sel = labels == name
            tt = outer.t[sel]
            used = np.unique(tt)
            remap = -np.ones(len(outer.v), dtype=np.int64)
            remap[used] = np.arange(len(used))
            p = Part("%s_%s" % (outer.name, name), outer.shape, outer.v[used], remap[tt], uv_of[name](used), name,
                     outer.mat, outer.w[used], outer.D[used])
            p.normals = mk_geom.normalize(outer.attrs["vn"][used])
            out.append(p)
        return out
    return split


# ---------------------------------------------------------------- weights and sliders
def cleft_mask(v, top=73.4):
    """The buttock cleft behind, bridged fully up to z 72.4 and less and less up to the waistband's top (the Wrap's
    old mask stopped at z 70.5: the back's top sagged 0.3-0.4 into the cleft, and the large bodies deepened it into a
    crease). The top of the waistband, under the loincloth's belt, dips a little into the cleft: bridged right up to
    the top, the middle came within 0.01-0.06 of the belt (which also spans the cleft) and crossed it in a crouch."""
    return np.clip((3.5 - np.abs(v[:, 0])) / 1.5, 0, 1) * np.clip((-3.0 - v[:, 1]) / 1.5, 0, 1) * \
        np.clip((top - v[:, 2]) / 1.0, 0, 1)


def bridged_proxy(proxy, iters=120):
    return W.bridged_proxy(proxy, iters)


def build_parts(body, proxy, log=print):
    env = W.pelvis_envelope(proxy)
    rings = ring_frames(env)
    v, t = cloth_surface(env, rings)
    v, t = cut_outline(v, t, rings)
    # the vulva changes most under the body sliders (labia, mons): a little more room over the skin weighted to its
    # bones, fading out smoothly (one extreme preset pushed it 0.18 through the briefs)
    loc, _, face, _ = mk_blend.nearest(body.tree, v)
    tri = body.t[face]
    bc = mk_blend.barycentric(loc, body.v[tri[:, 0]], body.v[tri[:, 1]], body.v[tri[:, 2]])
    k = np.einsum("nk,nk->n", bc, body.mask_bones(INTIMATE)[tri])[:, None]
    k = mk_geom.smooth(np.clip(k / 0.25, 0, 1), mk_geom.neighbor_pairs(t, len(v)), 12, 0.5)[:, 0]
    v = v + mk_geom.vertex_normals(v, t) * (0.2 * k * k * (3 - 2 * k))[:, None]
    v, t, d, trim = split_trim(v, t)
    log("briefs: %d verts, %d tris (%d in the trim), %d boundary loop(s)"
        % (len(v), len(t), trim.sum(), len(mk_geom.boundary_loops(t))))
    # weights: the skin's under every vertex, the cleft bridged across the midline, at most 4 bones. The waistband
    # lies under the loincloth's belt and takes the belt's own weights (the proxy's, smoothed): with the skin's (the
    # cleft's walls, the buttocks') it left the belt in a crouch and stood out over it at the back
    w = mk_fit.beneath_weights(v, t, body, None, smooth_iters=3)
    wb, _, _ = mk_fit.transfer(v, proxy, t, 4)
    th = np.arctan2(v[:, 0], v[:, 1] - Y_SPLIT)
    kw = smoothstep((v[:, 2] - (waist(th) - 1.3)) / 0.8)
    w = w * (1 - kw)[:, None] + wb * kw[:, None]
    w = mk_fit.blend_midline(w, body.bones, v[:, 0], 1.4)
    w, rep = mk_fit.limit_influences(w, t)
    log("briefs weights: cut %.3f -> %.3f (%d steps), largest jump across an edge %.3f"
        % (rep["cut_before"], rep["cut_after"], rep["steps"], mk_fit.edge_jumps(w, t).max()))
    # slider data: from the proxy with the cleft and crotch bridged; near the edges from the real skin (the elastic
    # follows it), except over the cleft at the back, whose walls sink in as the buttocks grow and pulled the cloth
    # into a crease down the middle
    gb = W.garment_bones(proxy.bones)
    _, D = bind(v, t, W.bridged_proxy(proxy), smooth_iters=4, bone_mask=gb)
    _, De, _ = mk_fit.transfer(v, body, None, 0, bone_filter=gb)
    # (the gusset is narrow: there only its edges, in the grooves beside the crotch, follow the real skin)
    crotch = smoothstep((2.0 - np.abs(v[:, 0])) / 0.8) * smoothstep((66.0 - v[:, 2]) / 1.0)
    k = smoothstep(mk_geom.edge_distance(v, t) / (1.6 - 1.0 * crotch))
    k = np.maximum(k, smoothstep((2.2 - np.abs(v[:, 0])) / 1.2) * smoothstep((-4.0 - v[:, 1]) / 1.5))
    D = De * (1 - k)[:, None, None] + D * k[:, None, None]
    inner = np.clip((mk_geom.edge_distance(v, t) - 0.8) / 1.2, 0, 1)
    D, _ = mk_fit.fold_safe_diffs(v, t, D, proxy.slider_names, skip=mk_export.NO_GARMENT,
                                  damp=inner * inner * (3 - 2 * inner), samples=96, rounds=6)
    # across the cleft (down to the perineum) the cloth spans the buttocks: its slider data smoothed sideways there
    cm = smoothstep((2.5 - np.abs(v[:, 0])) / 1.5) * smoothstep((-2.0 - v[:, 1]) / 1.5)
    pairs = mk_geom.neighbor_pairs(t, len(v))
    rows = np.nonzero(cm > 0)[0]
    flat = D.reshape(len(v), -1).astype(np.float64)
    for _ in range(60):
        flat[rows] += 0.5 * cm[rows, None] * mk_geom.laplacian(flat, pairs)[rows]
    D = flat.reshape(D.shape)
    # in front of the body under every slider. The pushes spread smoothly over the cloth (per vertex they left lumps
    # on large bodies) and go along the cloth's own normals (along the body's, which turn every way round the vulva,
    # the grooves beside it and the cleft, they crumpled the gusset and creased the middle); the real skin only counts
    # where the cloth lies on it (over the bridged cleft its walls pulled the cloth into a crease)
    bp = bridged_proxy(proxy)
    cn = soft_normals(v, t, 6)
    D = mk_fit.garment_conform(v, D, bp, normals=cn, smooth_tris=t, smooth_iters=40, min_cos=0.65)
    D = mk_fit.garment_conform(v, D, body, slack_margin=0.45, normals=cn, smooth_tris=t, smooth_iters=40,
                               min_cos=0.65)
    labels, uv_of = island_layout(v, t, d, trim, rings)
    parts = solid_parts("briefs", "Briefs", v, t, v[:, [0, 2]], w, D, THICK,
                        ("briefs", "patch:lining", "patch:edge"), ("linen", "lining", "edge"), rim_segments=2,
                        split_outer=island_splitter(labels, uv_of))
    for k, (c, n, up, along, on_env) in enumerate(rings):
        rv, rt, ruv = W.torus(c, n, up, RING_R, RING_WIRE, 32, 8)
        rw, rD = W.rigid_bind(rv, proxy, on_env)
        parts.append(Part("hipring%d" % k, "BriefsGold", rv, rt, ruv, "patch:gold", "gold", rw, rD))
    return parts, (v, t, labels)


def parts(body, proxy, log=print):
    """The briefs' parts, built once per run (the Loincloth with Briefs and the Briefs share them)."""
    key = id(proxy)
    if key not in _CACHE:
        out, _ = build_parts(body, proxy, log)
        for p in out:
            p.bones = list(proxy.bones)
            p.sliders = list(proxy.slider_names)
            if p.mat in ("gold", "gem"):
                p.shape = "BriefsGold"
            elif p.name.endswith("_in"):
                p.shape = "BriefsLining"
            else:
                p.shape = "Briefs"
        _CACHE[key] = out
    return _CACHE[key]


def build(body, proxy, log=print):
    out = parts(body, proxy, log)
    log("briefs: %d parts" % len(out))
    return {"parts": list(out)}
