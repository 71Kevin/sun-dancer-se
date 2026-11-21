"""Eclipse Scythe (two-handed, battleaxe animations): an ebony haft with gold bands and two leather grips, an
eclipse at the head (an obsidian disc inside a gold corona of rays), a crescent blade of blackened steel with gold
filigree (texture) and a softly glowing amber edge, a gold finial and a sun pommel.

Coordinates follow vanilla weapons: the haft along +Y with the right-hand grip at the origin, the blade in the XY
plane sweeping toward -X, thin along Z."""
import numpy as np

import mk_geom

SPINE =[(-1.2, 50.6), (-8.0, 52.3), (-16.0, 52.7), (-24.0, 51.4), (-31.5, 48.3), (-38.5, 43.1), (-44.0, 36.1),
         (-48.0, 27.6), (-50.2, 19.4)]
EDGE = [(-47.7, 26.4), (-43.4, 33.3), (-37.6, 38.7), (-30.6, 42.3), (-22.8, 44.2), (-14.6, 44.7), (-7.2, 44.2),
        (-1.2, 43.0)]
SAMPLES = 10
EDGE_BAND = 0.9
STEP = 0.32
SPINE_T = 0.56
EDGE_T = 0.06
UV_SCALE = 0.6
UV_X = 52.0
UV_Y = 56.0
HAFT = (-49.0, 50.5)
HAFT_R = 0.86
GRIPS = [(-27.0, -14.0), (-6.0, 6.5)]
BANDS = [-48.0, -28.2, -12.8, -7.2, 7.7, 30.0, 44.5]
DISC_C = (0.9, 51.2)
DISC_R = 3.9
DISC_T = 2.0
RAYS = 20


def outline():
    pts = np.array(SPINE + EDGE)
    return mk_geom.catmull_rom(pts, SAMPLES, closed=True)


def edge_line():
    o = outline()
    k0 = (len(SPINE) - 1) * SAMPLES
    return o[k0:k0 + len(EDGE) * SAMPLES + 1]


def polygon_sdf(p, poly):
    from piece_wrap import polygon_sdf as f
    return f(p, poly)


def polyline_dist(p, line):
    d = np.full(len(p), np.inf)
    for a, b in zip(line[:-1], line[1:]):
        ab = b - a
        t = np.clip(((p - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
        d = np.minimum(d, np.linalg.norm(p - (a + t[:, None] * ab), axis=1))
    return d


def half_thickness(p):
    k = np.clip(polyline_dist(p, edge_line()) / 5.0, 0, 1)
    return 0.5 * (EDGE_T + (SPINE_T - EDGE_T) * np.sqrt(k))


def plate_uv(p):
    return np.stack([p[:, 0] + UV_X, UV_Y - p[:, 1]], 1) * UV_SCALE


def blade_plate():
    o = outline()
    lo = o.min(0) - 0.6
    hi = o.max(0) + 0.6
    xs = np.arange(lo[0], hi[0] + 1e-6, STEP)
    ys = np.arange(lo[1], hi[1] + 1e-6, STEP)
    X, Y = np.meshgrid(xs, ys)
    v = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], 1)
    t = mk_geom.grid_tris(len(ys), len(xs))
    v, t = mk_geom.iso_cut(v, t, -polygon_sdf(v[:, :2], o))[:2]
    f = EDGE_BAND - polyline_dist(v[:, :2], edge_line())
    band = drop_crumbs(*mk_geom.iso_cut(v, t, f)[:2])
    body = drop_crumbs(*mk_geom.iso_cut(v, t, -f)[:2])
    return body, band


def drop_crumbs(v, t, min_tris=40):
    """Removes tiny disconnected slivers the cuts leave at the thin tip."""
    e = mk_geom.unique_edges(t)
    lab = mk_geom.components(len(v), e[:, 0], e[:, 1])
    tl = lab[t[:, 0]]
    cnt = np.bincount(tl, minlength=len(v))
    keep = cnt[tl] >= min_tris
    v2, t2, _, _ = mk_geom.compact(v, t[keep])
    return v2, t2


def extrude(v, t, open_edge=None):
    p = v[:, :2]
    h = half_thickness(p)
    n = len(v)
    top = np.c_[p, h]
    bot = np.c_[p, -h]
    tris = [t, t[:, ::-1] + n]
    rim_v, rim_t = [], []
    base = 2 * n
    for loop in mk_geom.boundary_loops(t):
        q = p[loop]
        tang = mk_geom.normalize(np.roll(q, -1, 0) - np.roll(q, 1, 0))
        out = np.stack([tang[:, 1], -tang[:, 0]], 1)
        mid = np.c_[q + out * (0.45 * h[loop])[:, None], np.zeros(len(loop))]
        ids = base + sum(len(r) for r in rim_v) + np.arange(len(loop))
        rim_v.append(mid)
        m = len(loop)
        for i in range(m):
            j = (i + 1) % m
            a, b = loop[i], loop[j]
            if open_edge is not None and open_edge[a] and open_edge[b]:
                continue
            rim_t += [[a, ids[i], b], [b, ids[i], ids[j]], [ids[i], a + n, ids[j]], [ids[j], a + n, b + n]]
    allv = np.concatenate([top, bot])
    plate = (allv, np.concatenate(tris), np.concatenate([plate_uv(p), plate_uv(p)]))
    if not rim_t:
        return plate, None
    rv = np.concatenate(rim_v)
    rt = np.array(rim_t)
    rim_all = np.concatenate([allv, rv])
    used = np.unique(rt)
    remap = -np.ones(len(rim_all), dtype=np.int64)
    remap[used] = np.arange(len(used))
    return plate, (rim_all[used], remap[rt], rim_all[used][:, [0, 2]])


def ring(y, radius, tube, segs=36):
    path = np.array([[np.cos(a) * radius, y, np.sin(a) * radius] for a in np.linspace(0, 2 * np.pi, segs + 1)])
    v, t, uv, _ = mk_geom.sweep(path, np.full(len(path), tube), 8, up=(0, 1, 0))
    return v, t, uv


def disc(center, radius, thick, seg=64, rings_=6):
    """Obsidian eclipse disc facing +-Z, slightly domed, with a bevelled rim."""
    c = np.asarray(center, dtype=np.float64)
    ang = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    verts = [c + [0, 0, thick * 0.62]]
    uv = [[0, 0]]
    for r in range(1, rings_ + 1):
        rr = radius * r / rings_
        z = thick * 0.5 + thick * 0.12 * (1 - (r / rings_) ** 2)
        for a in ang:
            verts.append(c + [np.cos(a) * rr, np.sin(a) * rr, z])
            uv.append([np.cos(a) * rr, np.sin(a) * rr])
    top = np.array(verts)
    bot = top * np.array([1, 1, -1]) + np.array([0, 0, 2 * c[2]])
    tri = [[0, 1 + k, 1 + (k + 1) % seg] for k in range(seg)]
    for r in range(1, rings_):
        a0, b0 = 1 + (r - 1) * seg, 1 + r * seg
        for k in range(seg):
            k1 = (k + 1) % seg
            tri += [[a0 + k, b0 + k, b0 + k1], [a0 + k, b0 + k1, a0 + k1]]
    tri = np.array(tri)
    n = len(top)
    edge = 1 + (rings_ - 1) * seg + np.arange(seg)
    side = []
    for k in range(seg):
        a, b = edge[k], edge[(k + 1) % seg]
        side += [[a, a + n, b + n], [a, b + n, b]]
    v = np.vstack([top, bot])
    t = np.vstack([tri, tri[:, ::-1] + n, np.array(side)])
    return v, t, np.vstack([uv, uv])


def corona(center, r0, r1, thick, rays=RAYS):
    """Gold rays around the disc: flattened pyramids in the disc plane, alternating long and short."""
    c = np.asarray(center, dtype=np.float64)
    vs, ts, uvs = [], [], []
    off = 0
    for k in range(rays):
        a = 2 * np.pi * (k + 0.5) / rays
        L = r1 if k % 2 == 0 else r0 + (r1 - r0) * 0.55
        d = np.array([np.cos(a), np.sin(a), 0.0])
        s = np.array([-np.sin(a), np.cos(a), 0.0])
        w = 0.55 if k % 2 == 0 else 0.42
        base = c + d * (r0 - 0.2)
        pts = [base + s * w + [0, 0, thick], base - s * w + [0, 0, thick], base - s * w - [0, 0, thick],
               base + s * w - [0, 0, thick], c + d * L]
        tri = [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4], [0, 3, 2], [0, 2, 1]]
        vs.append(np.array(pts))
        ts.append(np.array(tri) + off)
        uvs.append(np.array([[0, 0], [w, 0], [w, thick], [0, thick], [w * 0.5, L]]))
        off += 5
    return np.vstack(vs), np.vstack(ts), np.vstack(uvs)


def spike(base_y, length, radius, up=True):
    ys = np.linspace(0, 1, 14)
    sgn = 1 if up else -1
    path = np.stack([np.zeros_like(ys), base_y + sgn * ys * length, np.zeros_like(ys)], 1)
    r = radius * (1 - ys) ** 0.9 + 0.02
    v, t, uv, _ = mk_geom.sweep(path, r, 12, up=(0, 0, 1), cap_start=True, cap_end=True)
    if not up:
        t = t[:, ::-1].copy()
    return v, t, uv


def sun_pommel(y, radius):
    from piece_wrap import sun_disc
    v, t, uv = sun_disc(np.array([0.0, y, 0.0]), np.array([0, 0, 1.0]), np.array([0, 1.0, 0]), radius, 0.7, rays=12,
                        ray_len=radius * 0.45, rings=4)
    return v, t, uv


def build(body=None, proxy=None, log=print):
    from mk_parts import Part   # Blender only (the texture painter imports this module for the blade geometry)
    parts = []
    (bv, bt), (ev, et) = blade_plate()

    def seam(v):
        return np.abs(polyline_dist(v[:, :2], edge_line()) - EDGE_BAND) < 1e-3
    (pv, pt, puv), rim = extrude(bv, bt, open_edge=seam(bv))
    (gv, gt, guv), grim = extrude(ev, et, open_edge=seam(ev))
    parts.append(Part("blade", "ScytheBlade", pv, pt, puv, "blade", "steel"))
    parts.append(Part("blade_edge", "ScytheGlow", gv, gt, guv, "blade_edge", "ember"))
    if rim is not None:
        parts.append(Part("blade_rim", "ScytheBlade", rim[0], rim[1], rim[2], "patch:steel", "steel"))
    if grim is not None:
        parts.append(Part("blade_edge_rim", "ScytheGlow", grim[0], grim[1], grim[2], "patch:ember", "ember"))
    # haft
    # the haft in four texture islands (one 100-unit strip would set the scale of the whole atlas)
    bounds = np.linspace(HAFT[0], HAFT[1], 5)
    for k in range(4):
        ys = np.linspace(bounds[k], bounds[k + 1], 24)
        r = HAFT_R * (1 + 0.06 * np.cos(np.pi * (ys - HAFT[0]) / (HAFT[1] - HAFT[0])))
        path = np.stack([np.zeros_like(ys), ys, np.zeros_like(ys)], 1)
        hv, ht, huv, _ = mk_geom.sweep(path, r, 16, up=(0, 0, 1), cap_start=(k == 0), cap_end=(k == 3))
        huv = huv * [2 * np.pi * HAFT_R, 1.0] + [0, bounds[k] - HAFT[0]]
        parts.append(Part("haft%d" % k, "ScytheHaft", hv, ht, huv, "haft%d" % k, "wood"))
    # leather grips: a flat strap wound around the haft
    for gk, (y0, y1) in enumerate(GRIPS):
        pitch, width, thick = 0.62, 0.5, 0.06
        rr = HAFT_R + 0.05
        turns = (y1 - y0) / pitch
        a = np.linspace(0, 2 * np.pi * turns, int(turns * 20) + 2)
        gp = np.stack([np.cos(a) * rr, y0 + a / (2 * np.pi) * pitch, np.sin(a) * rr], 1)

        def prof(s, ang):
            return width * thick / np.sqrt((thick * np.cos(ang)) ** 2 + (width * np.sin(ang)) ** 2)
        wv, wt, wuv, _ = mk_geom.sweep(gp, np.ones(len(gp)), 6, profile=prof, up=(0, 1, 0))
        parts.append(Part("grip%d" % gk, "ScytheHaft", wv, wt, wuv, "patch:strap", "leather"))
    for k, y in enumerate(BANDS):
        rv, rt, ruv = ring(y, HAFT_R + 0.12, 0.2)
        parts.append(Part("band%d" % k, "ScytheGold", rv, rt, ruv, "patch:gold", "gold"))
    # socket collar where the blade meets the haft
    yy = np.linspace(41.2, 47.6, 14)
    sp = np.stack([np.zeros_like(yy), yy, np.zeros_like(yy)], 1)
    sr = HAFT_R + 0.32 + 0.12 * np.sin(np.linspace(0, np.pi, len(yy)))
    sv, st, suv, _ = mk_geom.sweep(sp, sr, 20, up=(0, 0, 1), cap_start=True, cap_end=True)
    parts.append(Part("socket", "ScytheGold", sv, st, suv, "patch:goldworn", "gold"))
    # the eclipse: obsidian disc, gold ring around it, corona of rays
    c = np.array([DISC_C[0], DISC_C[1], 0.0])
    dv, dt, duv = disc(c, DISC_R, DISC_T)
    parts.append(Part("disc", "ScytheDisc", dv, dt, duv, "disc", "obsidian"))
    path = np.array([c + [np.cos(a) * (DISC_R + 0.12), np.sin(a) * (DISC_R + 0.12), 0]
                     for a in np.linspace(0, 2 * np.pi, 73)])
    tv, tt, tuv, _ = mk_geom.sweep(path, np.full(len(path), 0.36), 10, up=(0, 0, 1))
    parts.append(Part("disc_ring", "ScytheGold", tv, tt, tuv, "patch:gold", "gold"))
    cv, ct, cuv = corona(c, DISC_R + 0.3, DISC_R + 2.3, 0.22)
    parts.append(Part("corona", "ScytheGold", cv, ct, cuv, "patch:gold", "gold"))
    fv, ft, fuv = spike(HAFT[1] + 4.6, 5.0, 0.55, up=True)
    parts.append(Part("finial", "ScytheGold", fv, ft, fuv, "patch:gold", "gold"))
    pv2, pt2, puv2 = sun_pommel(HAFT[0] - 0.9, 1.25)
    parts.append(Part("pommel", "ScytheGold", pv2, pt2, puv2, "patch:gold", "gold"))
    sv2, st2, suv2 = spike(HAFT[0] - 1.4, 3.2, 0.42, up=False)
    parts.append(Part("pommel_spike", "ScytheGold", sv2, st2, suv2, "patch:gold", "gold"))
    enlarge(parts)
    for p in parts:
        p.bones, p.w, p.D, p.sliders = [], None, None, []
    lo = np.min([p.v.min(0) for p in parts], axis=0)
    hi = np.max([p.v.max(0) for p in parts], axis=0)
    log("scythe: %d parts, %d verts, %.1f long, blade %.1f wide" % (len(parts), sum(len(p.v) for p in parts),
                                                                      hi[1] - lo[1], hi[0] - lo[0]))
    return {"parts": parts}


# the grips stay where the battleaxe animations put the hands; above the upper grip the haft is longer and the head
# (blade, eclipse, finial) is larger, below the lower grip the haft is a little longer
HEAD_SCALE = 1.3
HAFT_UP = 13.0      # the head moves up by this much (plus its own growth)
HAFT_DOWN = 4.0
HEAD_FROM = 40.6    # everything above this belongs to the head (socket and up)


HEAD_PARTS = ("blade", "socket", "disc", "corona", "finial")


def enlarge(parts):
    """Bigger presence without moving the hands: the head (blade, socket, eclipse, finial) grows about the socket's
    foot and rises on a longer haft, the pommel end drops a little. The haft stretches only along its length (its
    radius stays), between the grips nothing changes."""
    pivot = np.array([0.0, HEAD_FROM, 0.0])
    top_grip = GRIPS[1][1] + 1.0
    low_grip = GRIPS[0][0] - 1.0

    def along(y):
        out = y.copy()
        up = y > HEAD_FROM
        out[up] = HEAD_FROM + (y[up] - HEAD_FROM) * HEAD_SCALE + HAFT_UP
        mid = (y > top_grip) & ~up
        out[mid] = top_grip + (y[mid] - top_grip) * (HEAD_FROM + HAFT_UP - top_grip) / (HEAD_FROM - top_grip)
        low = y < low_grip
        out[low] = y[low] - HAFT_DOWN * np.clip((low_grip - y[low]) / 6.0, 0, 1)
        return out

    for p in parts:
        if p.name.startswith(HEAD_PARTS):
            p.v = pivot + (p.v - pivot) * HEAD_SCALE + np.array([0.0, HAFT_UP, 0.0])
        else:
            v = p.v.copy()
            v[:, 1] = along(p.v[:, 1])
            p.v = v
