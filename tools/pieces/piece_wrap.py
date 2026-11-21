"""Sun Dancer Wrap (body slot 32): linen triangle cups on an underbust band with a sun medallion, halter straps
knotted at the nape, a back clasp, and the UBE body with the skin under the cups and band trimmed away.

The briefs are not part of the Wrap: they are worn with the Loincloth or on their own (piece_briefs, which uses the
pelvis envelope and the bridged proxy below), over the Wrap's whole lower body, so the loincloth can also be worn
without them."""
import numpy as np

import mk_blend
import mk_export
import mk_fit
import mk_geom
import mk_preset
from mk_parts import Part, bind, solid_parts

CUP_OFF = 0.2
CUP_THICK = 0.09
# rounded-triangle cup outline in the breast chart (s: outward from the cleavage, t: up), units from the nipple
CUP_OUTLINE = [(-0.25, 6.3), (-1.85, 3.5), (-3.4, 0.5), (-3.95, -1.6), (-3.7, -3.4), (-1.8, -4.35), (0.6, -4.55),
               (2.9, -4.3), (4.4, -3.2), (4.75, -0.7), (3.7, 2.4), (1.6, 4.9)]
BAND = (89.15, 90.45)
BAND_OFF = 0.13
BAND_THICK = 0.1
STRAP_W = 0.46
STRAP_T = 0.085
STRAP_OFF = 0.17
HALTER = [(-3.75, 4.6, 103.2), (-4.55, 2.2, 106.6), (-4.75, -0.6, 109.0), (-3.9, -3.6, 110.0), (-2.2, -5.5, 110.3)]
INTIMATE = ("Vagina", "Pussy", "Clitoral", "Anus", "Anal", "Labia", "Clit")
# (Round 4 tried keeping the skin under the briefs in a partition of its own, hidden by a second slot on the items
# with the briefs, as UBE does with partition 53. Rejected: every free-looking slot is used in the player's load
# order by things worn every day (facial piercings in 55, chest piercings in 53, the handheld lantern in 61, other
# panties in 52): they would have taken the briefs off or opened a hole in the body. The briefs are worn over it.)


# ---------------------------------------------------------------- body helpers
def landmarks(v, body=None):
    """The breast apexes (front-most points; the cups are laid out from them) and the navel. The navel is found from
    the UBE navel slider when the body is given (centre of its displacement); otherwise as the deepest dimple of
    the belly's midline (the deepest point relative to the skin around it)."""
    front = v[:, 1] > 0
    nip = []
    for side in (-1, 1):
        m = front & (np.sign(v[:, 0]) == side) & (v[:, 2] > 85) & (v[:, 2] < 102) & (np.abs(v[:, 0]) < 12)
        nip.append(v[np.argmax(np.where(m, v[:, 1], -99))])
    names = getattr(body, "slider_names", ())
    navel = None
    for s in names:
        if s.lower().startswith("navel"):
            mag = np.linalg.norm(body.D[:, names.index(s)], axis=1)
            sel = mag > 0.25 * mag.max()
            if sel.sum() >= 4:
                navel = (v[sel] * mag[sel, None]).sum(0) / mag[sel].sum()
                navel = v[np.argmin(np.linalg.norm(v - navel, axis=1) + 4.0 * (v[:, 1] - navel[1]).clip(0))]
            break
    if navel is None:
        cand = np.nonzero(front & (np.abs(v[:, 0]) < 0.8) & (v[:, 2] > 70) & (v[:, 2] < 84))[0]
        depth = np.zeros(len(cand))
        for k, i in enumerate(cand):
            d = np.linalg.norm(v - v[i], axis=1)
            ring = front & (d > 1.2) & (d < 2.4)
            depth[k] = v[ring, 1].mean() - v[i, 1] if ring.any() else -9
        navel = v[cand[np.argmax(depth)]]
    return np.array(nip), navel


def fibonacci_dirs(n):
    k = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * k / n)
    th = np.pi * (1 + 5 ** 0.5) * k
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


def enclosed_vertices(body, mask, n_dirs=48, reach=4.0):
    """Vertices from which no ray escapes within `reach`: the inner walls of the vulva and anus cavities."""
    idx = np.nonzero(mask)[0]
    esc = np.zeros(len(idx), dtype=np.int64)
    for d in fibonacci_dirs(n_dirs):
        o = body.v[idx] + d * 0.01
        _, _, _, dist = mk_blend.raycast(body.tree, o, np.tile(d, (len(idx), 1)), reach)
        esc += ~np.isfinite(dist)
    out = np.zeros(len(body.v), dtype=bool)
    out[idx[esc == 0]] = True
    return out


def make_proxy(body):
    """Outer garment proxy: the body with navel, vulva and butt cleft smoothed over, the cavity walls removed and their
    openings capped, so garments bridge those details. The breasts are left as they are: the UBE nipple barely rises
    from the breast (the height follows the dome), and smoothing it cut the breast tip into a flat plateau whose
    slider data lagged behind the growing breast (the cups dented in the middle and raised a knob); the nipple and
    areola sliders are left out of garments anyway."""
    nip, navel = landmarks(body.v, body)
    v = body.v
    sm = body
    mask = np.zeros(len(v))
    d = np.linalg.norm(v - navel, axis=1)
    mask = np.maximum(mask, np.clip((2.4 - d) / 0.8, 0, 1))
    crotch = (np.abs(v[:, 0]) < 3.4) & (v[:, 2] > 59) & (v[:, 2] < 70.0) & (v[:, 1] > -6.0)
    mask = np.maximum(mask, crotch * np.clip((3.4 - np.abs(v[:, 0])) / 1.2, 0, 1))
    cleft = (np.abs(v[:, 0]) < 2.6) & (v[:, 2] > 57) & (v[:, 2] < 74.0) & (v[:, 1] < -3.5)
    mask = np.maximum(mask, cleft * np.clip((2.6 - np.abs(v[:, 0])) / 1.0, 0, 1))
    region = (np.abs(v[:, 0]) < 4) & (v[:, 2] > 58) & (v[:, 2] < 74)
    inner = enclosed_vertices(body, region)
    sm = sm.smoothed(mask * ~inner, iters=150)
    return capped(sm, inner)


def pelvis_envelope(proxy, voxel=0.22):
    """Watertight copy of the pelvis (voxel remesh of the proxy) with the vulva and the slit between the labia
    closed and ironed flat: the briefs are shaped on it, so they lie smooth over the crotch."""
    v, t = proxy.v, proxy.t
    caps = []
    extra = []
    normals = mk_geom.vertex_normals(v, t)
    for loop in mk_geom.boundary_loops(t):   # close the neck, wrists and ankles: the remesh needs a closed volume
        k = len(v) + len(extra)
        c = v[loop].mean(0)
        extra.append(c)
        f = np.stack([loop, np.roll(loop, -1), np.full(len(loop), k)], 1)
        out = mk_geom.normalize(c - v.mean(0))
        if np.dot(np.cross(v[f[:, 1]] - c, v[f[:, 0]] - c).sum(0), out) < 0:
            f = f[:, [1, 0, 2]]
        caps.append(f)
    if extra:
        v = np.vstack([v, np.array(extra)])
        t = np.vstack([t] + caps)
    ev, et = mk_blend.voxel_union([(v, t)], voxel)
    e = mk_geom.unique_edges(et)
    lab = mk_geom.components(len(ev), e[:, 0], e[:, 1])
    big = np.bincount(lab[et[:, 0]]).argmax()
    et = et[lab[et[:, 0]] == big]
    keep = (ev[et][:, :, 2].min(1) > 50.0) & (ev[et][:, :, 2].max(1) < 80.0)
    ev, et, _, _ = mk_geom.compact(ev, et[keep])
    if mk_geom.vertex_normals(ev, et)[np.argmax(ev[:, 1]), 1] < 0:
        et = et[:, ::-1].copy()
    k = np.clip((3.4 - np.abs(ev[:, 0])) / 1.4, 0, 1) * np.clip((69.5 - ev[:, 2]) / 1.5, 0, 1) * \
        np.clip((ev[:, 2] - 58.0) / 1.5, 0, 1) * np.clip((ev[:, 1] + 5.0) / 1.5, 0, 1)
    pairs = mk_geom.neighbor_pairs(et, len(ev))
    for _ in range(140):
        ev = ev + (0.5 * k)[:, None] * mk_geom.laplacian(ev, pairs)
    ev = mk_fit.fill_concave(ev, et, k, iters=80)
    return mk_fit.layer_body(ev, et, np.zeros((len(ev), 1, 3), dtype=np.float32))


def capped(proxy, inner):
    t = proxy.t[~inner[proxy.t].any(1)]
    v, w, D = proxy.v, proxy.w, proxy.D
    normals = mk_geom.vertex_normals(v, t)
    nv, nw, nD, fans = [], [], [], []
    for loop in mk_geom.boundary_loops(t):
        c = v[loop].mean(0)
        if abs(c[0]) > 3.0 or not 58.0 < c[2] < 74.0:
            continue
        k = len(v) + len(nv)
        nv.append(c)
        nw.append(w[loop].mean(0))
        nD.append(D[loop].mean(0))
        f = np.stack([loop, np.roll(loop, -1), np.full(len(loop), k)], 1)
        fn = np.cross(v[f[:, 1]] - c, v[f[:, 0]] - c).sum(0)
        if np.dot(fn, normals[loop].sum(0)) > 0:
            f = f[:, [1, 0, 2]]
        fans.append(f)
    cp = object.__new__(type(proxy))
    cp.__dict__.update(proxy.__dict__)
    if nv:
        cp.v = np.vstack([v, np.array(nv)])
        cp.t = np.vstack([t] + fans)
        cp.w = np.vstack([w, np.array(nw)])
        cp.D = np.concatenate([D, np.array(nD, dtype=D.dtype)])
    else:
        cp.t = t
    cp.n = mk_geom.vertex_normals(cp.v, cp.t)
    cp.tree = mk_blend.bvh(cp.v, cp.t)
    cp.removed = int(inner.sum())
    cp.caps = len(nv)
    return cp


def polygon_sdf(p, poly):
    poly = np.asarray(poly, dtype=np.float64)
    d = np.full(len(p), np.inf)
    inside = np.zeros(len(p), dtype=bool)
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        ab = b - a
        t = np.clip(((p - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
        d = np.minimum(d, np.linalg.norm(p - (a + t[:, None] * ab), axis=1))
        cross = (a[1] > p[:, 1]) != (b[1] > p[:, 1])
        xi = a[0] + (p[:, 1] - a[1]) * ab[0] / (ab[1] if ab[1] != 0 else 1e-12)
        inside ^= cross & (p[:, 0] < xi)
    return np.where(inside, -d, d)


def surface_path(proxy, ctrl, off, samples=10, iters=3):
    """Polyline through control points that lies on the proxy surface at `off`, with the surface normals."""
    p = mk_geom.catmull_rom(np.asarray(ctrl, dtype=np.float64), samples)
    p = mk_geom.resample_polyline(p, len(p))
    for _ in range(iters):
        loc, nrm, _, _ = mk_blend.nearest(proxy.tree, p)
        p = loc + nrm * off
        for _ in range(6):   # straighten the path along the surface (ends fixed)
            p[1:-1] += 0.5 * (0.5 * (p[:-2] + p[2:]) - p[1:-1])
        p = mk_geom.resample_polyline(p, len(p))
    loc, nrm, _, _ = mk_blend.nearest(proxy.tree, p)
    return loc + nrm * off, nrm


def ribbon(path, normals, width, thick, segs=8, caps=True):
    """Flat strap along a path: an elliptical cross section, wide across the surface and thin along its normal."""
    path = np.asarray(path, dtype=np.float64)
    tang = mk_geom.normalize(np.gradient(path, axis=0))
    n = mk_geom.normalize(normals - tang * (normals * tang).sum(1, keepdims=True))
    b = np.cross(n, tang)
    ang = np.linspace(0, 2 * np.pi, segs, endpoint=False)
    rings = [path[i] + np.outer(np.cos(ang), b[i]) * width * 0.5 + np.outer(np.sin(ang), n[i]) * thick * 0.5
             for i in range(len(path))]
    v = np.concatenate(rings)
    t = mk_geom.grid_tris(len(path), segs, wrap=True)
    L = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    uv = np.stack([np.tile(np.cos(ang) * width * 0.5, len(path)), np.repeat(L, segs)], 1)
    if caps:
        c0, c1 = len(v), len(v) + 1
        v = np.vstack([v, path[0], path[-1]])
        uv = np.vstack([uv, [0, 0], [0, L[-1]]])
        last = (len(path) - 1) * segs
        cap = [[c0, (k + 1) % segs, k] for k in range(segs)] + [[c1, last + k, last + (k + 1) % segs]
                                                                  for k in range(segs)]
        t = np.vstack([t, np.array(cap)])
    return v, t, uv


def rigid_bind(points, proxy, center=None):
    c = np.asarray(points).mean(0) if center is None else np.asarray(center)
    w, D, _ = mk_fit.transfer(c[None], proxy)
    return np.repeat(w, len(points), 0), np.repeat(D, len(points), 0)


def torus(center, normal, up, major, minor, seg_major=20, seg_minor=8):
    n = mk_geom.normalize(np.asarray(normal, dtype=np.float64))
    u = mk_geom.normalize(np.asarray(up, dtype=np.float64) - n * np.dot(up, n))
    w_ = np.cross(n, u)
    verts, uv = [], []
    for i in range(seg_major):
        a = 2 * np.pi * i / seg_major
        c = np.cos(a) * u + np.sin(a) * w_
        for j in range(seg_minor):
            b = 2 * np.pi * j / seg_minor
            verts.append(center + c * (major + minor * np.cos(b)) + n * (minor * np.sin(b)))
            uv.append([i / seg_major * 2 * np.pi * major, j / seg_minor * 2 * np.pi * minor])
    tris = []
    for i in range(seg_major):
        for j in range(seg_minor):
            a0 = i * seg_minor + j
            a1 = ((i + 1) % seg_major) * seg_minor + j
            b0 = i * seg_minor + (j + 1) % seg_minor
            b1 = ((i + 1) % seg_major) * seg_minor + (j + 1) % seg_minor
            tris.append([a0, a1, b1])
            tris.append([a0, b1, b0])
    return np.array(verts), np.array(tris), np.array(uv)


def sun_disc(center, normal, up, radius, thick, rays=16, ray_len=0.38, rings=5):
    """Sun medallion: a domed disc with a ring of pointed rays, extruded to `thick`."""
    n = mk_geom.normalize(np.asarray(normal, dtype=np.float64))
    u = mk_geom.normalize(np.asarray(up, dtype=np.float64) - n * np.dot(up, n))
    w_ = np.cross(n, u)
    m = rays * 4
    ang = np.linspace(0, 2 * np.pi, m, endpoint=False)
    k = np.arange(m) % 4
    r_out = np.where(k == 0, radius + ray_len, np.where(k == 2, radius + 0.04, radius + ray_len * 0.35))
    outline = np.stack([np.cos(ang) * r_out, np.sin(ang) * r_out], 1)
    pts2 = [np.zeros((1, 2))]
    for r in range(1, rings + 1):
        pts2.append(outline * (r / rings))
    pts2 = np.concatenate(pts2)
    rad = np.linalg.norm(pts2, axis=1)
    dome = 0.35 * thick * np.clip(1 - (rad / radius) ** 2, 0, 1)
    tris = []
    for j in range(m):
        tris.append([0, 1 + j, 1 + (j + 1) % m])
    for r in range(1, rings):
        a0, b0 = 1 + (r - 1) * m, 1 + r * m
        for j in range(m):
            j1 = (j + 1) % m
            tris += [[a0 + j, b0 + j, b0 + j1], [a0 + j, b0 + j1, a0 + j1]]
    tris = np.array(tris)
    top = center[None] + np.outer(pts2[:, 0], w_) + np.outer(pts2[:, 1], u) + np.outer(thick * 0.5 + dome, n)
    bot = center[None] + np.outer(pts2[:, 0], w_) + np.outer(pts2[:, 1], u) - n * thick * 0.5
    nv = len(top)
    side = []
    edge = 1 + (rings - 1) * m + np.arange(m)
    for j in range(m):
        a, b = edge[j], edge[(j + 1) % m]
        side += [[a, b + nv, b], [a, a + nv, b + nv]]
    v = np.vstack([top, bot])
    t = np.vstack([tris[:, ::-1], tris + nv, np.array(side)])
    uv = np.vstack([pts2, pts2 * 0.98])
    return v, t, uv


def dome(center, normal, radius, height, seg=16, rings=6):
    n = mk_geom.normalize(np.asarray(normal, dtype=np.float64))
    a = mk_geom.normalize(np.cross(n, [0, 0, 1.0]) if abs(n[2]) < 0.9 else np.cross(n, [1.0, 0, 0]))
    b = np.cross(n, a)
    verts, uv = [center + n * height], [[0, 0]]
    for r in range(1, rings + 1):
        th = (r / rings) * np.pi / 2
        for s in range(seg):
            ph = 2 * np.pi * s / seg
            p = center + (a * np.cos(ph) + b * np.sin(ph)) * radius * np.sin(th) + n * height * np.cos(th)
            verts.append(p)
            uv.append([np.cos(ph) * np.sin(th) * radius, np.sin(ph) * np.sin(th) * radius])
    tris = [[0, 1 + s, 1 + (s + 1) % seg] for s in range(seg)]
    for r in range(1, rings):
        a0, b0 = 1 + (r - 1) * seg, 1 + r * seg
        for s in range(seg):
            s1 = (s + 1) % seg
            tris += [[a0 + s, b0 + s, b0 + s1], [a0 + s, b0 + s1, a0 + s1]]
    return np.array(verts), np.array(tris)[:, ::-1].copy(), np.array(uv)


# ---------------------------------------------------------------- cups
def breast_frame(body, nip, side):
    near = np.linalg.norm(body.v - nip, axis=1) < 1.5
    f = mk_geom.normalize(body.n[near].mean(0))
    z = np.array([0, 0, 1.0])
    u = mk_geom.normalize(z - f * f.dot(z))
    o = np.array([side * 1.0, 0, 0])
    o = mk_geom.normalize(o - f * f.dot(o) - u * u.dot(o))
    return f, u, o


def cup_outline():
    return mk_geom.catmull_rom(np.array(CUP_OUTLINE), 10, closed=True)


def cup_surface(body, proxy, side):
    nip, _ = landmarks(body.v, body)
    n = nip[0 if side < 0 else 1]
    f, u, o = breast_frame(body, n, side)
    d = proxy.v - n
    s, t, h = d @ o, d @ u, d @ f
    sdf = polygon_sdf(np.stack([s, t], 1), cup_outline())
    cand = (sdf < 1.5) & (h > -4.0) & (np.sign(proxy.v[:, 0]) == side)
    tri = proxy.t[cand[proxy.t].all(1)]
    v, tt, _, used = mk_geom.compact(proxy.v, tri)
    e = mk_geom.unique_edges(tt)
    lab = mk_geom.components(len(v), e[:, 0], e[:, 1])
    seed = int(np.argmin(np.linalg.norm(v - n, axis=1)))
    keep = lab[tt[:, 0]] == lab[seed]
    v, tt, _, used2 = mk_geom.compact(v, tt[keep])
    used = used[used2]
    nrm = mk_geom.vertex_normals(v, tt)
    p = v + nrm * CUP_OFF
    attrs = {"s": s[used], "t": t[used]}
    p, tt, attrs = mk_geom.iso_cut(p, tt, -sdf[used], attrs)
    p = mk_fit.drape(p, tt, proxy, CUP_OFF, iters=12, lam=0.35)
    p = mk_geom.smooth_boundary(p, tt, iters=6, lam=0.4)
    p = mk_fit.push_out(p, proxy, CUP_OFF, iters=2)
    return p, tt, attrs, (n, f, u, o)


def cup_parts(body, proxy, side):
    v, t, a, frame = cup_surface(body, proxy, side)
    w, D = bind(v, t, proxy, smooth_iters=3, edge_body=body)
    # the cup follows the breast's own slider data (the proxy keeps the breasts unsmoothed) and keeps its distance
    # along the breast's new normal when big breast sliders turn the surface (otherwise the lower cup sank in)
    D = mk_fit.garment_conform(v, mk_fit.normal_turn_diffs(v, D, proxy), proxy)
    # linen stretched over the breast: away from its edges the cup's slider data is smoothed at the scale of a
    # nipple, so the small bumps some breast sliders raise on the skin do not show through; the skin under the
    # inside of the cup is trimmed, and the edges (where the kept skin meets the cup) keep the exact data
    k = np.clip((mk_geom.edge_distance(v, t) - 0.8) / 1.2, 0, 1)[:, None, None]
    Ds = mk_geom.smooth(D.reshape(len(v), -1), mk_geom.neighbor_pairs(t, len(v)), 30, 0.5).reshape(D.shape)
    D = (D * (1 - k) + Ds * k).astype(np.float32)
    uv = mk_blend.unwrap(v, t)
    tag = "L" if side < 0 else "R"
    parts = solid_parts("cup" + tag, "Wrap", v, t, uv, w, D, CUP_THICK, ("cup" + tag, "patch:lining", "patch:edge"),
                        ("linen", "lining", "edge"), rim_segments=2)
    apex = v[np.argmax(a["t"])]
    return parts, frame, apex, (v, t, a, w, D)


def bead(center, radius, seg=8, rings=6):
    verts, uv = [], []
    for r in range(rings + 1):
        ph = np.pi * r / rings
        for s in range(seg):
            th = 2 * np.pi * s / seg
            verts.append(center + radius * np.array([np.sin(ph) * np.cos(th), np.sin(ph) * np.sin(th), np.cos(ph)]))
            uv.append([th * radius, ph * radius])
    t = mk_geom.grid_tris(rings + 1, seg, wrap=True)
    v = np.array(verts)
    if ((v - center) * mk_geom.vertex_normals(v, t)).sum() < 0:
        t = t[:, ::-1].copy()
    return v, t, np.array(uv)


FRINGE_T = -1.9     # the fringe's trim runs across the lower front of the cup, this far below the nipple


def surface_body(v, t, w, D):
    """A garment surface wrapped as a body (signed distance, projection, slider data) for push_out and
    garment_conform: things hung on the garment stay in front of it for every body."""
    sb = object.__new__(mk_fit.Body)
    sb.v, sb.t = np.asarray(v, dtype=np.float64), np.asarray(t)
    sb.w, sb.D = np.asarray(w), np.asarray(D, dtype=np.float32)
    sb.n = mk_geom.vertex_normals(sb.v, sb.t)
    sb.tree = mk_blend.bvh(sb.v, sb.t)
    return sb


def _lerp(arr, i, j, k):
    arr = np.asarray(arr)
    kk = k.reshape((-1,) + (1,) * (arr.ndim - 1))
    return arr[i] * (1 - kk) + arr[j] * kk


def iso_line(v, t, f, arrays, level=0.0):
    """Points where the scalar f crosses `level` on the mesh edges, with `arrays` interpolated there."""
    e = mk_geom.unique_edges(t)
    fa, fb = f[e[:, 0]] - level, f[e[:, 1]] - level
    cross = (fa * fb) < 0
    e, fa, fb = e[cross], fa[cross], fb[cross]
    k = fa / (fa - fb)
    return _lerp(v, e[:, 0], e[:, 1], k), [_lerp(a, e[:, 0], e[:, 1], k) for a in arrays]


def _resample(L, s, arr):
    """Values of a polyline attribute (given at cumulative lengths L) at the lengths s."""
    i = np.clip(np.searchsorted(L, s), 1, len(L) - 1)
    k = np.clip((s - L[i - 1]) / np.maximum(L[i] - L[i - 1], 1e-9), 0, 1)
    return _lerp(arr, i - 1, i, k)


def fringe_parts(proxy, cup, side, spacing=0.55, beads=5, pitch=0.3):
    """A dancer's fringe on the cup: a fine gold chain across the lower front of the cup (below the breast's
    apex) and, hanging from it, strands of small gold and amber beads, each ending in a sun coin, longer in the
    middle. It used to hang from the cup's lower edge, under the breast, where a large breast covered it from the
    front; hung from the front of the cup it falls in front of the breast's lower curve on every body. A strand is
    rigid on the cup (it swings with the breast physics); its slider data keeps it in front of the cup, the ribs and
    the belly for every body."""
    from ev_shapes import chain_links, coin
    v, t, a, w, D = cup
    n = mk_geom.vertex_normals(v, t)
    pts, (s_at, w_at, D_at, n_at) = iso_line(v, t, a["t"], [a["s"], w, D, n], FRINGE_T)
    if len(pts) < 4:
        return []
    order = np.argsort(s_at)
    pts, w_at, D_at, n_at = pts[order], w_at[order], D_at[order], mk_geom.normalize(n_at[order])
    for _ in range(6):     # the crossing points zig-zag a little across the triangles
        pts[1:-1] = 0.5 * pts[1:-1] + 0.25 * (pts[:-2] + pts[2:])
    L = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    end = 0.3              # clear of the cup's rolled edge at both ends
    cupb = surface_body(v, t, w, D)
    down = np.array([0, 0, -1.0])
    tag = "L" if side < 0 else "R"
    out = []

    def keep_in_front(c, D0):
        D1 = mk_fit.garment_conform(c[None], D0[None].astype(np.float64), proxy)
        return mk_fit.garment_conform(c[None], D1, cupb)

    # the trim: a fine chain lying on the cup along the line
    sl = np.linspace(end, L[-1] - end, max(int((L[-1] - 2 * end) / 0.1), 2))
    path = _resample(L, sl, pts) + _resample(L, sl, n_at) * 0.05
    lv, lt, luv, along = chain_links(path, link=0.22, width=0.13, wire=0.026)
    la = sl[0] + along * (sl[-1] - sl[0])
    lw = _resample(L, la, w_at)
    lD = _resample(L, la, D_at)
    out.append(Part("fringetrim" + tag, "WrapGold", lv, lt, luv, "patch:gold", "gold", lw, lD.astype(np.float32)))
    # the strands (the line runs from the inner end, by the cleavage, outward): none in the first stretch by the
    # cleavage, where large breasts press together over them
    inner = 1.2
    stations = np.arange(inner + spacing * 0.5, L[-1] - end - spacing * 0.3, spacing)
    for k, s in enumerate(stations):
        p0 = _resample(L, np.array([s]), pts)[0]
        n0 = mk_geom.normalize(_resample(L, np.array([s]), n_at)[0])
        w0 = _resample(L, np.array([s]), w_at)[0]
        D0 = _resample(L, np.array([s]), D_at)[0]
        u = (s - inner) / max(L[-1] - inner - end, 1e-6)
        length = beads * pitch * (0.78 + 0.42 * (1 - abs(2 * u - 1)) + 0.06 * np.sin(k * 1.7))
        top = p0 + n0 * 0.09
        centers = top[None] + np.outer((np.arange(beads) + 0.6) * length / beads, down)
        centers = mk_fit.push_out(centers, proxy, 0.22, iters=3)
        centers = mk_fit.push_out(centers, cupb, 0.1, iters=3)
        pieces = []
        for j, c in enumerate(centers):
            bv, bt, buv = bead(c, 0.075 if j % 2 == 0 else 0.065)
            pieces.append(("fringe%s%d_%d" % (tag, k, j), bv, bt, buv, "patch:gold" if j % 2 == 0 else "patch:amber",
                           "gold" if j % 2 == 0 else "gem", c))
        last = centers[-1]
        face = mk_geom.normalize(n0 - down * np.dot(n0, down))
        cc = mk_fit.push_out((last + down * 0.32)[None], proxy, 0.12, iters=3)
        cc = mk_fit.push_out(cc, cupb, 0.08, iters=3)[0]
        cv, ct, cuv = coin(cc, face, [0, 0, 1.0], radius=0.22, thick=0.04)
        pieces.append(("fringecoin%s%d" % (tag, k), cv, ct, cuv, "coin", "gold", cc))
        for name, pv, pt, puv, island, mat, c in pieces:
            pD = keep_in_front(c, D0)
            out.append(Part(name, "WrapGold", pv, pt, puv, island, mat, np.repeat(w0[None], len(pv), 0),
                            np.repeat(pD, len(pv), 0)))
    return out


# ---------------------------------------------------------------- band, medallion, clasp, halter
def band_parts(proxy):
    zs = np.arange(BAND[0] - 0.45, BAND[1] + 0.46, 0.15)
    cy = mk_fit.torso_axis(proxy, zs)
    p, t, theta, zz = mk_fit.ring_grid(proxy, zs, 200, cy)
    p = mk_fit.drape(p, t, proxy, BAND_OFF, iters=30, lam=0.4)
    rows, cols = len(zs), 200
    g = p.reshape(rows, cols, 3)
    ring = np.linalg.norm(np.diff(np.concatenate([g, g[:, :1]], 1), axis=1), axis=2)
    S = np.concatenate([np.zeros((rows, 1)), np.cumsum(ring, 1)[:, :-1]], 1).ravel()
    th = np.arctan2(np.sin(theta), np.cos(theta))
    a = {"S": S, "z0": zz, "theta": th}
    first = np.arange(rows) * cols
    p, t, a = mk_geom.open_ring_seam(p, t, first, a, "S", np.cumsum(ring, 1)[:, -1])
    v, tt, a = mk_geom.iso_cut(p, t, BAND[1] - a["z0"], a)
    v, tt, a = mk_geom.iso_cut(v, tt, a["z0"] - BAND[0], a)
    w, D = bind(v, tt, proxy, smooth_iters=3)
    D = mk_fit.garment_conform(v, D, proxy)
    uv = np.stack([a["S"], a["z0"] - BAND[0]], 1)
    from ev_shapes import island_splitter
    parts = solid_parts("band", "Wrap", v, tt, uv, w, D, BAND_THICK, ("band", "patch:lining", "patch:edge"),
                        ("leather", "lining", "edge"), rim_segments=2, attrs={"S": a["S"]},
                        split_outer=island_splitter("S", [0.5 * a["S"].max()], "band"))
    tree = mk_blend.bvh(v, tt)
    zc = 0.5 * (BAND[0] + BAND[1])
    out = {}
    for key, y0, dy in (("front", -2.0, 1.0), ("back", 2.0, -1.0)):
        loc, nrm, _, _ = mk_blend.raycast(tree, np.array([[0.0, y0, zc]]), np.array([[0, dy, 0.0]]), 30.0)
        out[key] = (loc[0], nrm[0] if np.dot(nrm[0], [0, dy, 0]) > 0 else -nrm[0])
    return parts, out, (v, tt, w, D)


def band_fringe(proxy, band_surface, half_width=6.6, spacing=0.78):
    """Sun coins on fine chains along the front of the underbust band (a dancer's coin fringe, visible whatever the
    breast size; the fringe under the cups hides under large breasts). Each coin is rigid on the band's edge above
    it and kept in front of the belly for every body."""
    from ev_shapes import chain_links, coin
    v, t, w, D = band_surface
    wv, wt, first, inv = mk_fit.welded(v, t)
    loops = mk_geom.boundary_loops(wt)
    lower = min(loops, key=lambda lp: wv[lp][:, 2].mean())
    pts = wv[lower]
    sel = (pts[:, 1] > 0) & (np.abs(pts[:, 0]) < half_width)
    order = np.argsort(pts[sel][:, 0])
    edge = pts[sel][order]
    ids = first[np.asarray(lower)[sel][order]]
    L = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(edge, axis=0), axis=1))])
    out = []
    for k, s in enumerate(np.arange(spacing * 0.5, L[-1], spacing)):
        i = int(np.clip(np.searchsorted(L, s), 1, len(L) - 1))
        f = (s - L[i - 1]) / max(L[i] - L[i - 1], 1e-9)
        p0 = edge[i - 1] * (1 - f) + edge[i] * f
        vi = ids[i] if f > 0.5 else ids[i - 1]
        loc, nrm, _, _ = mk_blend.nearest(proxy.tree, p0[None])
        n = mk_geom.normalize(nrm[0] - np.array([0, 0, 1.0]) * nrm[0][2])
        length = 0.3 if k % 2 else 0.62
        top = p0 + n * 0.06
        bottom = mk_fit.push_out((top + np.array([0, 0, -length]))[None], proxy, 0.16, iters=3)[0]
        lv, lt, luv, _ = chain_links([top, bottom], link=0.2, width=0.12, wire=0.025)
        cc = mk_fit.push_out((bottom + np.array([0, 0, -0.3]))[None], proxy, 0.14, iters=3)[0]
        cv, ct, cuv = coin(cc, n, [0, 0, 1.0], radius=0.26, thick=0.045)
        cw = w[vi][None]
        for name, pv, pt, puv, island, c in (("bandchain%d" % k, lv, lt, luv, "patch:gold", top),
                                             ("bandcoin%d" % k, cv, ct, cuv, "coin", cc)):
            pD = mk_fit.garment_conform(c[None], D[vi][None].astype(np.float64), proxy)
            out.append(Part(name, "WrapGold", pv, pt, puv, island, "gold", np.repeat(cw, len(pv), 0),
                            np.repeat(pD, len(pv), 0)))
    return out


def medallion_parts(anchor, proxy):
    c, n = anchor
    up = np.array([0, 0, 1.0])
    c0 = c + n * 0.2
    dv, dt, duv = sun_disc(c0, n, up, 1.05, 0.2, rays=16)
    gv, gt, guv = dome(c0 + n * 0.17, n, 0.42, 0.22)
    rv, rt, ruv = torus(c0 + n * 0.12, n, up, 0.52, 0.06, 28, 6)
    w, D = rigid_bind(np.vstack([dv]), proxy, c)
    out = [Part("medallion", "WrapGold", dv, dt, duv, "medallion", "gold", w, D)]
    w2, D2 = rigid_bind(gv, proxy, c)
    out.append(Part("medallion_gem", "WrapGold", gv, gt, guv, "patch:amber", "gem", w2, D2))
    w3, D3 = rigid_bind(rv, proxy, c)
    out.append(Part("medallion_ring", "WrapGold", rv, rt, ruv, "patch:gold", "gold", w3, D3))
    return out


def clasp_parts(anchor, proxy):
    c, n = anchor
    up = np.array([0, 0, 1.0])
    side = mk_geom.normalize(np.cross(up, n))
    path = np.array([c + n * 0.16 - side * 0.75, c + n * 0.19, c + n * 0.16 + side * 0.75])
    path = mk_geom.catmull_rom(path, 6)
    norms = np.tile(n, (len(path), 1))
    v, t, uv = ribbon(path, norms, 1.25, 0.16, segs=10)
    w, D = rigid_bind(v, proxy, c)
    out = [Part("clasp", "WrapGold", v, t, uv, "patch:gold", "gold", w, D)]
    gv, gt, guv = dome(c + n * 0.27, n, 0.28, 0.14)
    w2, D2 = rigid_bind(gv, proxy, c)
    out.append(Part("clasp_gem", "WrapGold", gv, gt, guv, "patch:amber", "gem", w2, D2))
    return out


def halter_parts(proxy, apexes):
    """The halter straps from the cups round the neck, meeting at the nape in a small flat knot (under the Collar's
    torque when it is worn). The earlier ring knot with two tails hanging down the back read, together with the old
    collar's back pendant, as a necklace worn back to front."""
    parts = []
    # the knot sits in the middle of the Collar torque's slim back (z 109.7 to 110.9 at the nape), under it
    nape = np.array([0.0, -5.75, 110.15])
    for side, apex in zip((-1, 1), apexes):
        ctrl = [apex + np.array([0, 0, 0.15])] + [np.array([abs(x) * side, y, z]) for x, y, z in HALTER] + \
            [nape + np.array([side * 0.15, 0, 0])]
        path, nrm = surface_path(proxy, ctrl, STRAP_OFF, samples=12)
        v, t, uv = ribbon(path, nrm, STRAP_W, STRAP_T, segs=8)
        w, D = bind(v, t, proxy, smooth_iters=2)
        D = mk_fit.garment_conform(v, D, proxy)
        parts.append(Part("halter" + ("L" if side < 0 else "R"), "Wrap", v, t, uv, "patch:strap", "leather", w, D))
    loc, nrm, _, _ = mk_blend.nearest(proxy.tree, nape[None])
    c = loc[0] + nrm[0] * (STRAP_OFF + STRAP_T)
    kv, kt, kuv = dome(c, nrm[0], 0.34, 0.08, seg=16, rings=4)
    w, D = rigid_bind(kv, proxy, loc[0])
    parts.append(Part("halter_knot", "Wrap", kv, kt, kuv, "patch:strap", "leather", w, D))
    return parts


# ---------------------------------------------------------------- helpers for the briefs (piece_briefs)
def bridged_proxy(proxy, iters=120):
    """The proxy with its slider data smoothed over the buttock cleft and between the thighs: what a garment spanning
    them should follow."""
    v = proxy.v
    cleft = np.clip((4.5 - np.abs(v[:, 0])) / 1.5, 0, 1) * np.clip((-2.0 - v[:, 1]) / 1.5, 0, 1)
    legs = np.clip((6.0 - np.abs(v[:, 0])) / 2.0, 0, 1) * np.clip((67.0 - v[:, 2]) / 2.0, 0, 1)
    k = np.maximum(cleft, legs) * np.clip((v[:, 2] - 50.0) / 2.0, 0, 1) * np.clip((74.0 - v[:, 2]) / 2.0, 0, 1)
    pairs = mk_geom.sub_pairs(mk_geom.neighbor_pairs(proxy.t, len(v)), k > 0)
    rows = pairs[3]
    D = proxy.D.reshape(len(v), -1).astype(np.float64)
    for _ in range(iters):
        D[rows] += 0.5 * k[rows, None] * mk_geom.laplacian(D, pairs)[rows]
    bp = object.__new__(type(proxy))
    bp.__dict__.update(proxy.__dict__)
    bp.D = D.reshape(proxy.D.shape).astype(np.float32)
    return bp


def garment_bones(bones):
    return np.array([0.0 if any(k in b for k in INTIMATE) else 1.0 for b in bones])


# ---------------------------------------------------------------- skin trim
def covered_mask(body_v, body_n, cover_parts, out_dist=4.0, in_dist=1.6):
    vs, ts = [], []
    off = 0
    for p in cover_parts:
        vs.append(p.v)
        ts.append(p.t + off)
        off += len(p.v)
    tree = mk_blend.bvh(np.concatenate(vs), np.concatenate(ts))
    loc, nrm, face, dist = mk_blend.raycast(tree, body_v + body_n * 0.01, body_n, out_dist)
    outward = np.isfinite(dist) & ((nrm * body_n).sum(1) > 0)
    loc, nrm, face, dist = mk_blend.raycast(tree, body_v - body_n * 0.01, -body_n, in_dist)
    inward = np.isfinite(dist) & ((nrm * body_n).sum(1) > 0)
    return outward | inward


def trim_body(body, covered, rings=1, min_island=60):
    pairs = mk_geom.neighbor_pairs(body.t, len(body.v))
    free = ~covered
    lab = mk_geom.components(len(body.v), pairs[0], pairs[1], free)
    sizes = np.bincount(lab[free], minlength=len(body.v))
    keep = free & (sizes[lab] >= min_island)
    for _ in range(rings):
        grow = keep.copy()
        np.logical_or.at(grow, pairs[0], keep[pairs[1]])
        keep = grow
    tri = body.inv[body.orig_t["BaseShape"]]
    return ~(~keep)[tri].all(1)


def follow_skin(parts, body, skin_tris, near=0.75, far=1.5, mats=("linen", "leather", "lining", "edge"),
                bone_mask=None):
    tree = mk_blend.bvh(body.v, skin_tris)
    bw = body.w if bone_mask is None else body.w * bone_mask[None, :]
    for p in parts:
        if p.mat not in mats:
            continue
        loc, nrm, face, dist = mk_blend.nearest(tree, p.v)
        tri = skin_tris[face]
        bc = mk_blend.barycentric(loc, body.v[tri[:, 0]], body.v[tri[:, 1]], body.v[tri[:, 2]])
        ws = np.einsum("nk,nkb->nb", bc, bw[tri])
        ws /= np.maximum(ws.sum(1, keepdims=True), 1e-9)
        k = np.clip((far - dist) / (far - near), 0, 1)
        k = k * k * (3 - 2 * k)
        p.w = p.w * (1 - k)[:, None] + ws * k[:, None]
        p.w /= np.maximum(p.w.sum(1, keepdims=True), 1e-9)


# ---------------------------------------------------------------- build
def build(body, proxy=None, log=print, conform=True, corpus=None):
    import time
    t0 = time.time()
    proxy = proxy or make_proxy(body)
    log("proxy %.1fs (cavity verts removed %s, caps %s)" % (time.time() - t0, getattr(proxy, "removed", "?"),
                                                           getattr(proxy, "caps", "?")))
    parts, apexes, fringe = [], [], []
    for side in (-1, 1):
        p, fr, apex, cup = cup_parts(body, proxy, side)
        parts += p
        apexes.append(apex)
        fringe += fringe_parts(proxy, cup, side)
    log("cups %.1fs, fringe %d parts" % (time.time() - t0, len(fringe)))
    bparts, anchors, band_surface = band_parts(proxy)
    parts += bparts
    fringe += band_fringe(proxy, band_surface)
    parts += medallion_parts(anchors["front"], proxy)
    parts += clasp_parts(anchors["back"], proxy)
    parts += halter_parts(proxy, apexes)
    log("band, medallion, clasp, halter %.1fs" % (time.time() - t0))
    parts += fringe
    for p in parts:
        p.bones = list(proxy.bones)
        p.sliders = list(proxy.slider_names)
    outer = [p for p in parts if p.mat in ("linen", "leather") and not p.name.endswith(("_in", "_rim"))
             and not p.name.startswith(("halter_knot", "halter_tail"))]
    covered = covered_mask(body.v, body.n, outer)
    keep_tris = trim_body(body, covered)
    tri = body.inv[body.orig_t["BaseShape"]]
    kept = np.zeros(len(body.v), dtype=bool)
    kept[np.unique(tri[keep_tris])] = True
    under = np.nonzero(kept & covered)[0]
    log("trim: %d of %d skin triangles kept %.1fs" % (keep_tris.sum(), len(keep_tris), time.time() - t0))
    D_conf = body.D
    if conform:
        uv_names = {a["name"] for a in body.uv_attrs}
        for p in outer:
            p.sliders = list(proxy.slider_names)
        D_conf, rows = mk_fit.conform_diffs(body, body.D, under, outer, uv_names)
        vs = np.concatenate([p.v for p in outer])
        ts = np.concatenate([p.t + o for p, o in zip(outer, np.cumsum([0] + [len(p.v) for p in outer[:-1]]))])
        _, _, _, dist = mk_blend.nearest(mk_blend.bvh(vs, ts), body.v)
        near = np.nonzero(kept & (dist < 1.2))[0]
        samples = [vals for _, vals in (corpus if corpus is not None else mk_preset.corpus(body.slider_attrs))]
        D_conf, hist = mk_fit.conform_samples(body, D_conf, tri[keep_tris], near, outer, samples,
                                              skip=mk_export.NO_GARMENT, rounds=20)
        under = np.union1d(under, near)
        log("conform %d skin verts over %d preset samples, pokes per sample %s %.1fs"
            % (len(under), len(samples), " ".join("%.1f" % h for h in hist), time.time() - t0))
    follow_skin(parts, body, tri[keep_tris])
    for p in parts:
        if p.mat in ("gold", "gem"):
            p.shape = "WrapGold"
        elif p.name.endswith("_in"):
            p.shape = "WrapLining"
        else:
            p.shape = "Wrap"
    return {"parts": parts, "proxy": proxy, "keep_tris": keep_tris, "D_conf": D_conf, "under": under}
