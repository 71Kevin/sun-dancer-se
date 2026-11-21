"""Shared shape builders for the Sun Dancer pieces: bands around limbs and torso, hanging panels, studs."""
import numpy as np

import mk_blend
import mk_fit
import mk_geom
from mk_parts import Part, bind, solid_parts
from piece_wrap import dome, rigid_bind, sun_disc, torus  # noqa: F401  (re-exported for the pieces)


def frame_for(axis):
    a = mk_geom.normalize(np.asarray(axis, dtype=np.float64))
    ref = np.array([0, 1.0, 0]) if abs(a[1]) < 0.9 else np.array([1.0, 0, 0])
    u = mk_geom.normalize(ref - a * a.dot(ref))
    w = np.cross(a, u)
    return a, u, w


def limb_grid(proxy, a, b, s0, s1, step=0.15, cols=96, reach=14.0, radius=6.0):
    """Surface grid around a limb: rings perpendicular to the a->b axis at distances s0..s1 from a, rays cast from
    the axis outwards (first hit = the skin). Returns points, normals, triangles (wrapped), ring parameter s and
    angle (0 = the u direction, i.e. the front when the axis is roughly vertical)."""
    ax, u, w = frame_for(np.asarray(b) - np.asarray(a))
    ss = np.arange(s0, s1 + 1e-6, step)
    th = np.linspace(0, 2 * np.pi, cols, endpoint=False)
    dirs = np.outer(np.cos(th), u) + np.outer(np.sin(th), w)
    # ray origins at the centre of the limb's cross-section (the joint axis can lie off-centre or even outside)
    centers = np.asarray(a)[None] + np.outer(ss, ax)
    for k, c in enumerate(centers):
        for rad in (radius, radius * 0.8):
            d = proxy.v - c
            h = d @ ax
            r = np.linalg.norm(d - np.outer(h, ax), axis=1)
            m = (np.abs(h) < 0.35) & (r < rad)
            if m.sum() > 8:
                q = proxy.v[m].mean(0)
                c = c + (q - c) - ax * np.dot(q - c, ax)
        centers[k] = c
    if len(centers) > 2:
        centers[1:-1] = (centers[:-2] + 2 * centers[1:-1] + centers[2:]) / 4
    O = np.repeat(centers, cols, 0)
    Dd = np.tile(dirs, (len(ss), 1))
    loc, nrm, face, dist = mk_blend.raycast(proxy.tree, O, Dd, reach)
    # rays that miss (past the open end of a limb, or through a gap): ring median radius, or the nearest good ring
    R = dist.reshape(len(ss), cols)
    good_ring = np.isfinite(R).mean(1) > 0.6
    if not good_ring.any():
        raise ValueError("limb_grid: no ring hits the surface")
    idx = np.arange(len(ss))
    near = idx[good_ring][np.argmin(np.abs(idx[:, None] - idx[good_ring][None]), axis=1)]
    for k in range(len(ss)):
        src = R[near[k]]
        med = np.nanmedian(np.where(np.isfinite(src), src, np.nan))
        row = np.where(np.isfinite(R[k]) & good_ring[k], R[k], np.where(np.isfinite(src), src, med))
        R[k] = row
    dist = R.ravel()
    miss = ~np.isfinite(loc[:, 0]) | np.repeat(~good_ring, cols)
    loc[miss] = O[miss] + Dd[miss] * dist[miss, None]
    nrm[miss] = Dd[miss]
    t = mk_geom.grid_tris(len(ss), cols, wrap=True)[:, ::-1].copy()
    return loc, nrm, t, np.repeat(ss, cols), np.tile(th, len(ss)), (ax, u, w), len(ss), cols


def axial_rig(proxy, a, b, s0, s1, keep=None, radius=3.6, step=0.25):
    """Weights for a rigid band around the limb a->b (see mk_fit.axial_profile): returns f(points) -> weights.
    keep: substrings of the bones the band may follow (others are dropped, e.g. the upper arm for a forearm guard)."""
    s_vals = np.arange(s0, s1 + step, step)
    filt = None
    if keep is not None:
        filt = np.array([1.0 if any(k in bn for k in keep) else 0.0 for bn in proxy.bones])
    table = mk_fit.axial_profile(proxy, a, b, s_vals, radius=radius, bone_filter=filt)

    def weights(points):
        return mk_fit.axial_weights(points, a, b, s_vals, table)
    return weights


def band(proxy, name, shape, a, b, s0, s1, off, thick, island, mat, cols=96, step=0.12, iters=24, rim=2,
         edge_fn=None, body=None, uv_scale=1.0, radius=6.0, weights_fn=None):
    """Closed band around a limb or the torso between s0 and s1 (optionally a custom inside function edge_fn(s,
    theta) >= 0 instead of the straight ends), draped at `off`, solidified, weights and sliders from the proxy.
    weights_fn(points): bone weights that replace the projected ones (rigid rings, see axial_rig)."""
    pad = 0.6
    p, n, t, s, th, fr, rows, cc = limb_grid(proxy, a, b, s0 - pad, s1 + pad, step, cols, radius=radius)
    p = p + n * off
    p = mk_fit.drape(p, t, proxy, off, iters=iters, lam=0.4)
    g = p.reshape(rows, cc, 3)
    ring = np.linalg.norm(np.diff(np.concatenate([g, g[:, :1]], 1), axis=1), axis=2)
    S = np.concatenate([np.zeros((rows, 1)), np.cumsum(ring, 1)[:, :-1]], 1).ravel()
    attrs = {"S": S, "s": s, "th": th}
    p, t, attrs = mk_geom.open_ring_seam(p, t, np.arange(rows) * cc, attrs, "S", np.cumsum(ring, 1)[:, -1])
    f_in = edge_fn(attrs["s"], attrs["th"]) if edge_fn is not None else np.minimum(attrs["s"] - s0, s1 - attrs["s"])
    v, tt, attrs = mk_geom.iso_cut(p, t, f_in, attrs)
    v = mk_geom.smooth_boundary(v, tt, iters=6, lam=0.4)
    v = mk_fit.push_out(v, proxy, off, iters=2)
    tt = facing_out(v, tt, proxy)
    w, D = bind(v, tt, proxy, smooth_iters=3, edge_body=body)
    if weights_fn is not None:
        w = weights_fn(v)
    D = mk_fit.garment_conform(v, D, proxy)
    uv = np.stack([attrs["S"], attrs["s"]], 1) * uv_scale
    parts = solid_parts(name, shape, v, tt, uv, w, D, thick, (island, "patch:lining", "patch:edge"),
                        (mat, "lining", "edge"), rim_segments=rim)
    return parts, (v, tt, w, D, attrs, fr)


def edge_tubes(v, tt, a, axis, thick, radius, samples=96, seg=8, smooth=6):
    """Rolled edges of a band round a limb (axis through a along `axis`): a tube along each of its two boundary
    loops. A band's cut edges are uneven (points at odd spacings, noisy mesh normals: tubes swept along them came
    out lumpy), so each loop is resampled evenly, smoothed, and sunk to the middle of the band's thickness along the
    limb's radial direction. Closed sweeps (no crease where a tube closes). Returns [(v, t, uv, idx)], idx: for each
    tube vertex, the band vertex (index into v) whose weights and slider data it takes."""
    wv, wt, first, inv = mk_fit.welded(v, tt)
    ax = mk_geom.normalize(np.asarray(axis, dtype=np.float64))
    a = np.asarray(a, dtype=np.float64)
    out = []
    for loop in sorted(mk_geom.boundary_loops(wt), key=len, reverse=True)[:2]:
        loop = np.asarray(loop)
        P = mk_geom.resample_polyline(wv[loop], samples, closed=True)
        for _ in range(smooth):
            P = 0.5 * P + 0.25 * (np.roll(P, 1, axis=0) + np.roll(P, -1, axis=0))
        cen = P.mean(0)          # the loop's own centre (the joint axis can run off-centre in the limb)
        rad = mk_geom.normalize(P - cen - np.outer((P - cen) @ ax, ax))
        path = P - rad * thick * 0.5
        tv, tt_, tuv, cols = mk_geom.sweep(np.vstack([path, path[:1]]), np.full(len(path) + 1, radius), seg,
                                           up=tuple(ax), closed=True)
        near = np.argmin(np.linalg.norm(P[:, None] - wv[loop][None], axis=2), axis=1)
        src = first[loop[near]]
        idx = np.repeat(np.concatenate([src, src[:1]]), cols)[:len(tv)]
        out.append((tv, tt_, tuv, idx))
    return out


def island_splitter(attr, cuts, base_island):
    """split_outer callback for solid_parts: cuts the outer layer at the given values of a vertex attribute into
    separate texture islands (long strips pack badly in the atlas). Lining and rim are left whole."""
    def split(outer):
        aa = dict(outer.attrs, w=outer.w, D=outer.D, uv=outer.uv)
        edges = [-np.inf] + list(cuts) + [np.inf]
        out = []
        for k in range(len(edges) - 1):
            lo, hi = edges[k], edges[k + 1]
            v, t, a = outer.v, outer.t, aa
            if np.isfinite(lo):
                v, t, a = mk_geom.iso_cut(v, t, a[attr] - lo, a)
            if np.isfinite(hi) and len(t):
                v, t, a = mk_geom.iso_cut(v, t, hi - a[attr], a)
            if len(t) == 0:
                continue
            p = Part("%s_%d" % (outer.name, k), outer.shape, v, t, a["uv"], "%s%d" % (base_island, k), outer.mat,
                     a["w"], a["D"])
            p.normals = mk_geom.normalize(a["vn"])
            p.attrs = {kk: vv for kk, vv in a.items() if kk not in ("w", "D", "uv", "vn")}
            out.append(p)
        return out
    return split


def facing_out(v, t, proxy):
    """Winding such that the faces look away from the body (the outer layer of solid_parts must face outwards)."""
    fc = v[t].mean(1)
    loc, _, _, _ = mk_blend.nearest(proxy.tree, fc)
    away = ((fc - loc) * mk_geom.face_normals(v, t)).sum(1)
    return t if (away > 0).mean() >= 0.5 else t[:, ::-1].copy()


def surface_point(tree_v_t, origin, direction, reach=30.0):
    tree = tree_v_t
    loc, nrm, _, dist = mk_blend.raycast(tree, np.asarray(origin, dtype=np.float64)[None],
                                         np.asarray(direction, dtype=np.float64)[None], reach)
    n = nrm[0] if np.dot(nrm[0], -np.asarray(direction)) > 0 else -nrm[0]
    return loc[0], n


def chain_links(path, link=0.3, width=0.17, wire=0.035, seg_major=6, seg_minor=3):
    """A fine chain along a polyline: oval links `link` long, each overlapping the next, every other one turned a
    quarter turn about the chain. Returns vertices, triangles, uv (for a uniform patch) and each vertex's position
    along the chain (0..1, for weights that change along it)."""
    path = np.asarray(path, dtype=np.float64)
    L = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    n = max(int(L / (link * 0.72)), 1)
    pts = mk_geom.resample_polyline(path, n + 1)
    ref = np.array([0.0, 0.0, 1.0])
    vs, ts, uvs, along = [], [], [], []
    off = 0
    ax, ay = link * 0.5 - wire, width * 0.5 - wire
    for k in range(n):
        a, b = pts[k], pts[k + 1]
        d = mk_geom.normalize(b - a)
        e = np.cross(d, ref if abs(d[2]) < 0.9 else np.array([1.0, 0, 0]))
        e = mk_geom.normalize(e)
        if k % 2:
            e = np.cross(d, e)
        f = np.cross(d, e)
        mid = 0.5 * (a + b)
        for i in range(seg_major):
            t = 2 * np.pi * i / seg_major
            c = mid + d * ax * np.cos(t) + e * ay * np.sin(t)
            radial = mk_geom.normalize(d * ay * np.cos(t) + e * ax * np.sin(t))
            for j in range(seg_minor):
                s = 2 * np.pi * j / seg_minor
                vs.append(c + radial * wire * np.cos(s) + f * wire * np.sin(s))
                uvs.append([i / seg_major, j / seg_minor])
                along.append((k + 0.5) / n)
        for i in range(seg_major):
            for j in range(seg_minor):
                a0 = off + i * seg_minor + j
                a1 = off + ((i + 1) % seg_major) * seg_minor + j
                b0 = off + i * seg_minor + (j + 1) % seg_minor
                b1 = off + ((i + 1) % seg_major) * seg_minor + (j + 1) % seg_minor
                ts += [[a0, a1, b1], [a0, b1, b0]]
        off += seg_major * seg_minor
    v, t = np.array(vs), np.array(ts)
    nrm = mk_geom.vertex_normals(v, t)
    if ((v - np.repeat(0.5 * (pts[:-1] + pts[1:]), seg_major * seg_minor, 0)) * nrm).sum() < 0:
        t = t[:, ::-1].copy()
    return v, t, np.array(uvs), np.array(along)


def catenary(p0, p1, sag, n=24):
    """A chain hanging between two points: a parabola sagging `sag` below the straight line at the middle."""
    s = np.linspace(0, 1, n)[:, None]
    return (1 - s) * np.asarray(p0, dtype=np.float64) + s * np.asarray(p1, dtype=np.float64) - \
        np.array([0, 0, 1.0]) * sag * 4 * s * (1 - s)


COIN_UV = 0.4      # uv radius of every coin (texture units)


def coin(center, normal, up, radius=0.34, thick=0.06, seg=16):
    """A small gold coin facing `normal` (both faces stamped: uv in the coin's own frame, so every coin shares one
    texture island), with a bevelled rim."""
    n = mk_geom.normalize(np.asarray(normal, dtype=np.float64))
    u = mk_geom.normalize(np.asarray(up, dtype=np.float64) - n * np.dot(up, n))
    w = np.cross(n, u)
    ang = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    rings = [0.0, 0.55, 0.88, 1.0]
    lift = [0.5, 0.42, 0.5, 0.3]
    vs, uv = [], []
    for sgn in (1, -1):
        vs.append(center + n * thick * 0.5 * sgn * lift[0] * 2)
        uv.append([0.0, 0.0])
        for r, h in zip(rings[1:], lift[1:]):
            for a in ang:
                p = (np.cos(a) * w + np.sin(a) * u) * radius * r
                vs.append(center + p + n * thick * h * sgn)
                # the same uv disc for every coin whatever its size (they share one stamped coin in the atlas)
                uv.append([np.cos(a) * r * COIN_UV * sgn, np.sin(a) * r * COIN_UV])
    per = 1 + seg * (len(rings) - 1)
    tris = []
    for face in (0, 1):
        o = face * per
        for k in range(seg):
            tris.append([o, o + 1 + k, o + 1 + (k + 1) % seg])
        for r in range(len(rings) - 2):
            a0, b0 = o + 1 + r * seg, o + 1 + (r + 1) * seg
            for k in range(seg):
                k1 = (k + 1) % seg
                tris += [[a0 + k, b0 + k, b0 + k1], [a0 + k, b0 + k1, a0 + k1]]
    t = np.array(tris)
    t[:len(t) // 2] = t[:len(t) // 2][:, ::-1]
    edge0 = 1 + (len(rings) - 2) * seg
    side = []
    for k in range(seg):
        k1 = (k + 1) % seg
        a, b = edge0 + k, edge0 + k1
        side += [[a, b + per, b], [a, a + per, b + per]]
    t = np.vstack([t, np.array(side)])
    v = np.array(vs)
    if ((v - center) * mk_geom.vertex_normals(v, t)).sum() < 0:
        t = t[:, ::-1].copy()
    return v, t, np.array(uv)


def teardrop(center, normal, down, radius, length, seg=14, rings=10):
    """A gem or gold drop hanging along `down`."""
    n = mk_geom.normalize(normal)
    d = mk_geom.normalize(down)
    s = np.cross(d, n)
    verts, uv = [], []
    for r in range(rings + 1):
        u = r / rings
        z = (u - 0.5) * length
        rad = radius * np.sin(np.pi * u) ** 0.8 * (0.55 + 0.45 * u)
        for k in range(seg):
            a = 2 * np.pi * k / seg
            verts.append(center + d * z + (s * np.cos(a) + n * np.sin(a) * 0.55) * rad)
            uv.append([a * radius, z])
    t = mk_geom.grid_tris(rings + 1, seg, wrap=True)
    return np.array(verts), t[:, ::-1].copy(), np.array(uv)


def stud(center, normal, up, radius, thick, kind="disc", rays=12):
    """Small ornament on a strap: a domed disc or a little sun (rays)."""
    if kind == "sun":
        return sun_disc(center, normal, up, radius, thick, rays=rays, ray_len=radius * 0.45, rings=3)
    v, t, uv = dome(center, normal, radius, thick, seg=18, rings=4)
    return v, t, uv


def hanging_panel(proxy, side, x_top, x_bot, z_top, z_bot, drop=0.8, off_top=0.45, off_bot=0.9, slope=0.07,
                  rows_step=0.2, cols=34, window=2.2):
    """A cloth panel hanging from z_top to z_bot in front of (side=+1) or behind (side=-1) the body: it touches the
    body where the body sticks out and hangs straight below, bridging the gap between the legs. Returns the grid
    points, triangles, the contact row of each column and the x/z grid coordinates."""
    zs = np.arange(z_top, z_bot - 1e-6, -rows_step)
    rows = len(zs)
    u = np.linspace(0, 1, cols)
    env = np.full((rows, cols), -np.inf)
    X = np.zeros((rows, cols))
    for r, z in enumerate(zs):
        k = (z_top - z) / (z_top - z_bot)
        x0 = x_top[0] + (x_bot[0] - x_top[0]) * k
        x1 = x_top[1] + (x_bot[1] - x_top[1]) * k
        X[r] = x0 + (x1 - x0) * u
    Z = np.repeat(zs[:, None], cols, 1)
    xs = np.linspace(min(x_top[0], x_bot[0]) - window, max(x_top[1], x_bot[1]) + window, 4 * cols)
    O = np.zeros((rows * len(xs), 3))
    O[:, 0] = np.tile(xs, rows)
    O[:, 2] = np.repeat(zs, len(xs))
    O[:, 1] = side * 40.0
    d = np.tile([0, -side * 1.0, 0], (len(O), 1))
    loc, nrm, face, dist = mk_blend.raycast(proxy.tree, O, d, 80.0)
    hit_y = np.where(np.isfinite(dist), loc[:, 1] * side, -np.inf).reshape(rows, len(xs))
    for r in range(rows):
        for c in range(cols):
            m = np.abs(xs - X[r, c]) <= window
            env[r, c] = hit_y[r, m].max() if m.any() else -np.inf
    # where no ray meets the body (between the ankles, under a long skirt) the cloth just hangs (an envelope of
    # -inf); only the top row must touch the body
    if not np.isfinite(env[0]).all():
        env[0][~np.isfinite(env[0])] = env[0][np.isfinite(env[0])].max()
    # bridge the gap between the legs: raise valleys across each row (an upper envelope, never lowered)
    for _ in range(60):
        sm = env.copy()
        sm[:, 1:-1] = 0.5 * (env[:, :-2] + env[:, 2:])
        env = np.maximum(env, sm)
    # clearance grows quickly below the belt (room for the hip crease when the thighs lift) and slowly further down
    off = off_top + (off_bot - off_top) * np.clip((z_top - Z) / (z_top - z_bot), 0, 1) ** 0.45
    Y = np.zeros((rows, cols))
    contact = np.zeros((rows, cols), dtype=bool)
    for r in range(rows):
        want = env[r] + off[r]
        if r == 0:
            Y[r] = want
            contact[r] = True
        else:
            hang = Y[r - 1] - slope * rows_step
            Y[r] = np.maximum(want, hang)
            contact[r] = want >= hang
    # cloth tension along the hang: no ripples where the panel alternates between touching and hanging
    for _ in range(40):
        sm = Y.copy()
        sm[1:-1] = 0.5 * (Y[:-2] + Y[2:])
        Y = np.maximum(Y, sm)
    # bottom hem: a shallow point in the middle
    Zb = Z - drop * np.clip(1 - np.abs(u - 0.5) * 2, 0, 1)[None, :] * ((z_top - Z) / (z_top - z_bot)) ** 4
    pts = np.stack([X, Y * side, Zb], -1).reshape(-1, 3)
    t = mk_geom.grid_tris(rows, cols)
    if side < 0:
        t = t[:, ::-1].copy()
    pairs = mk_geom.neighbor_pairs(t, len(pts))
    fixed = np.zeros(len(pts), dtype=bool)
    fixed[:cols] = True
    pts = mk_geom.taubin(pts, pairs, iters=14, fixed=fixed)
    return pts, t, contact.ravel(), X.ravel(), Zb.reshape(-1), rows, cols


def panel_bind(pts, t, proxy, body, contact, rows, cols, side, top_rows=4, smooth=10, bone_mask=None):
    """Weights and sliders of a hanging panel: rows touching the body take the skin right behind them; hanging rows
    take the weights and slider data of the last row that touched the body (they hang from it); the hanging cloth's
    weights are smoothed across the panel."""
    n = len(pts)
    d = np.tile([0, -side * 1.0, 0], (n, 1))
    loc, nrm, face, dist = mk_blend.raycast(proxy.tree, pts, d, 6.0)
    miss = ~np.isfinite(dist)
    if miss.any():
        l2, _, f2, _ = mk_blend.nearest(proxy.tree, pts[miss])
        loc[miss], face[miss] = l2, f2
    tri = proxy.t[face]
    bc = mk_blend.barycentric(loc, proxy.v[tri[:, 0]], proxy.v[tri[:, 1]], proxy.v[tri[:, 2]])
    bw = proxy.w if bone_mask is None else proxy.w * bone_mask[None, :]
    W = np.einsum("nk,nkb->nb", bc, bw[tri])
    D = np.einsum("nk,nksd->nsd", bc, proxy.D[tri])
    W = W.reshape(rows, cols, -1)
    D = D.reshape(rows, cols, D.shape[1], 3)
    C = contact.reshape(rows, cols)
    for c in range(cols):
        last = 0
        for r in range(rows):
            if C[r, c]:
                last = r
            else:
                # hanging cloth: weights and slider data of the row it hangs from (it moves with the belly or
                # buttocks it drapes over). Until round 5 the weights turned to the thighs behind it within ~2
                # units; with the physics chains (held to the pelvis) taking over a few units further down, that
                # band following the thighs bent the panel sideways into an S whenever the hips turned against
                # the legs
                W[r, c] = W[last, c]
                D[r, c] = D[last, c]
    W = W.reshape(n, -1)
    D = D.reshape(n, D.shape[2], 3)
    pairs = mk_geom.neighbor_pairs(t, n)
    # rows lying on the body keep exactly the skin's weights (they move like the briefs under them); only the
    # hanging cloth is smoothed, so the legs stretch it gradually
    fixed = contact.copy()
    W = mk_geom.smooth(W, pairs, iters=smooth * 2, lam=0.5, fixed=fixed)
    D = mk_geom.smooth(D.reshape(n, -1), pairs, iters=smooth, lam=0.5, fixed=fixed).reshape(D.shape)
    W /= np.maximum(W.sum(1, keepdims=True), 1e-9)
    return W, D
