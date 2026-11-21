"""Paints the Sun Dancer texture atlases and writes the 8K, 4K and 2K texture sets, vanilla and PBR.

  python tools/make_textures.py <work> <out_textures> [outfit|scythe ...]

For each atlas (outfit: SunDancer, scythe: EclipseScythe) the painting is done once at 8192 px from the mesh
manifest (islands, materials, baked ambient occlusion) and reduced for 4K and 2K:
  vanilla  <stem>.dds diffuse, <stem>_n.dds normal + specular in alpha, <stem>_m.dds environment mask,
           <stem>_g.dds glow (scythe)
  PBR      pbr\\<stem>.dds albedo, pbr\\<stem>_n.dds normal, pbr\\<stem>_rmaos.dds roughness/metallic/AO/specular,
           pbr\\<stem>_g.dds emissive (scythe) -- used by PG Patcher through the PBRNifPatcher JSON
Everything is painted procedurally; no texture comes from another mod."""
import math
import os
import pickle
import struct
import subprocess
import sys
import time

import cv2
import numpy as np

TOOLS = os.path.dirname(os.path.abspath(__file__))
TEXCONV = os.environ.get("MK_TEXCONV", r"C:\Modding\_tools\texconv\texconv.exe")
SETS = (("8K", 0), ("4K", 1), ("2K", 2))
F = np.float32
MARGIN = 12
AO_FACTOR = 4
THREAD = (214, 170, 84)
COIN_UV = 0.4      # uv radius of every coin (ev_shapes.COIN_UV)
t0 = time.time()


def log(msg):
    print("[%6.1fs] %s" % (time.time() - t0, msg), flush=True)


def rgb(c):
    return np.array(c, dtype=F) / 255.0


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def fract(x):
    return x - np.floor(x)


class Ctx:
    def __init__(self, U, V, mask, dist, ao, seed, scale, key):
        self.h, self.w = mask.shape
        self.U, self.V = U, V
        self.X, self.Y = U * F(scale), V * F(scale)
        self.mask, self.dist, self.ao = mask, dist, ao
        self.seed, self.scale, self.key = seed, scale, key
        self.rough = None
        self.metal = None
        self.env = None
        self.alpha = None      # opacity (sheer fabric); None = opaque
        self.glow = None


def value_noise(c, rng, cx, cy):
    gx = (c.X - c.X.min()) / F(cx)
    gy = (c.Y - c.Y.min()) / F(cy)
    gw, gh = int(gx.max()) + 2, int(gy.max()) + 2
    g = rng.standard_normal((gh + 1, gw + 1)).astype(F)
    xi = np.floor(gx)
    fx = gx - xi
    xi = xi.astype(np.int64)
    yi = np.floor(gy)
    fy = gy - yi
    yi = yi.astype(np.int64)
    fx = fx * fx * fx * (fx * (fx * 6 - 15) + 10)
    fy = fy * fy * fy * (fy * (fy * 6 - 15) + 10)
    a = g[yi, xi] + (g[yi, xi + 1] - g[yi, xi]) * fx
    b = g[yi + 1, xi] + (g[yi + 1, xi + 1] - g[yi + 1, xi]) * fx
    return a + (b - a) * fy


def noise(c, seed, cell, octaves=4, gain=0.5, su=1.0, sv=1.0):
    rng = np.random.default_rng(seed)
    out = np.zeros((c.h, c.w), F)
    amp, total = 1.0, 0.0
    for _ in range(octaves):
        out += F(amp) * value_noise(c, rng, max(cell * su, 1.0), max(cell * sv, 1.0))
        total += amp
        amp *= gain
        cell /= 2
    return out * F(1.4 / total)


def pebbles(c, cell, seed):
    rng = np.random.default_rng(seed)
    img = np.ones((c.h, c.w), np.uint8)
    n = max(1, int(c.h * c.w / (cell * cell)))
    img[rng.integers(0, c.h, n), rng.integers(0, c.w, n)] = 0
    _, lab = cv2.distanceTransformWithLabels(img, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    edge = np.ones((c.h, c.w), np.uint8)
    edge[:, 1:][lab[:, 1:] != lab[:, :-1]] = 0
    edge[1:, :][lab[1:, :] != lab[:-1, :]] = 0
    d = cv2.distanceTransform(edge, cv2.DIST_L2, 3)
    top = np.clip(d / (cell * 0.42), 0, 1)
    return 1.0 - (1.0 - top) ** 2


def full(c, v):
    return np.full((c.h, c.w), v, F)


# ------------------------------------------------------------------ materials
def linen_base(c):
    p = F(0.085)
    wu = np.sin(F(2 * np.pi) * c.U / p)
    wv = np.sin(F(2 * np.pi) * c.V / p)
    weave = 0.5 + 0.5 * wu * wv
    slub_u = noise(c, c.seed, 0.9 * c.scale, octaves=2, su=6.0, sv=0.12)
    slub_v = noise(c, c.seed + 1, 0.9 * c.scale, octaves=2, su=0.12, sv=6.0)
    mott = noise(c, c.seed + 2, 2.4 * c.scale)
    color = rgb((236, 229, 212))[None, None] * (0.9 + 0.1 * weave[..., None] + 0.035 * (slub_u + slub_v)[..., None]
                                                 + 0.025 * mott[..., None])
    height = 0.0018 * weave + 0.0008 * (slub_u + slub_v)
    spec = 0.05 + 0.02 * weave
    c.rough = 0.86 - 0.05 * weave
    c.metal = full(c, 0.0)
    return color, height, spec


def gold_thread_band(c, color, height, spec, d0, d1, motif="diamond", period=0.34):
    """Embroidered band between distances d0 and d1 from the garment edge: a bronze-brown ground with gold thread
    worked in a diamond lattice (or a chevron row), outlined by two gold lines."""
    d = c.dist
    inside = smoothstep(d0 - 0.02, d0 + 0.02, d) * smoothstep(d1 + 0.02, d1 - 0.02, d)
    ground = rgb((84, 54, 30))
    if motif == "diamond":
        a = np.abs(fract(c.U / F(period)) - 0.5) + np.abs(fract(c.V / F(period)) - 0.5)
        pat = smoothstep(0.32, 0.26, a) * (1 - smoothstep(0.16, 0.1, a) * 0.85)
    else:
        mid = 0.5 * (d0 + d1)
        w = (d1 - d0) * 0.5
        z = np.abs(fract((c.U + c.V) / F(period)) - 0.5) * 2
        pat = smoothstep(0.18, 0.08, np.abs((d - mid) / max(w, 1e-3) - (z - 0.5) * 1.2))
    lines = smoothstep(0.03, 0.0, np.abs(d - d0 - 0.035)) + smoothstep(0.03, 0.0, np.abs(d - d1 + 0.035))
    gold = np.clip(pat + lines, 0, 1) * inside
    tw = 0.5 + 0.5 * np.sin(F(2 * np.pi) * (c.U - c.V) / F(0.03))
    gcol = rgb(THREAD)[None, None] * (0.85 + 0.15 * tw[..., None])
    color = color * (1 - inside[..., None]) + (ground[None, None] * (1 - gold[..., None]) + gcol * gold[..., None]) \
        * inside[..., None]
    height = height + 0.004 * inside + 0.004 * gold
    spec = spec * (1 - inside) + (0.08 * (1 - gold) + 0.55 * gold) * inside
    c.rough = c.rough * (1 - inside) + (0.7 * (1 - gold) + 0.32 * gold) * inside
    c.metal = c.metal * (1 - inside) + 0.85 * gold * inside
    return color, height, spec


def linen(c):
    color, height, spec = linen_base(c)
    if c.key.startswith("panel"):
        color, height, spec = gold_thread_band(c, color, height, spec, 0.16, 0.95, "diamond", 0.38)
        vmax = float(c.V[c.mask > 0].max()) if (c.mask > 0).any() else 0.0
        hem = smoothstep(vmax - 3.1, vmax - 3.0, c.V) * smoothstep(vmax - 0.9, vmax - 1.0, c.V) * \
            (c.dist > 0.95)
        if hem.any():
            sun = np.abs(fract(c.U / F(1.6)) - 0.5) * 2
            ray = smoothstep(0.12, 0.05, np.abs(sun - smoothstep(vmax - 3.0, vmax - 1.0, c.V)))
            hemcol = rgb((96, 60, 30)) * (1 - ray[..., None]) + rgb(THREAD) * ray[..., None]
            color = color * (1 - hem[..., None]) + hemcol * hem[..., None]
            height = height + 0.004 * hem * ray
            spec = spec * (1 - hem) + (0.08 + 0.45 * ray) * hem
            c.rough = c.rough * (1 - hem) + (0.7 - 0.38 * ray) * hem
            c.metal = c.metal * (1 - hem) + 0.85 * ray * hem
    elif c.key.startswith("brieftrim"):
        # the briefs' trim strips are laid out by arc length (U) and distance from the edge (V): the embroidered
        # band (the cups' chevron) is as wide all along, and its inner edge is the seam to the plain cloth
        edge = c.dist
        c.dist = np.maximum(c.V, F(0))
        color, height, spec = gold_thread_band(c, color, height, spec, 0.05, 0.47, "chevron", 0.42)
        c.dist = edge
    elif c.key.startswith("briefs_"):
        pass    # the briefs' cloth: plain linen (the trim strips carry the embroidery)
    else:
        color, height, spec = gold_thread_band(c, color, height, spec, 0.1, 0.52, "chevron", 0.42)
    return color, height, spec


SAFFRON = (246, 190, 84)
SUNSET = (206, 104, 36)


def chiffon(c):
    """Sheer saffron chiffon (the arm veils): a fine open weave, warm at the gathered top and a deeper sunset amber
    toward the hem, soft long sheen. The alpha carries the sheerness (about half); the gold-embroidered border along
    the sides and the hem, and the rolled edge, are opaque. The top under the cord is left plain."""
    p = F(0.045)
    wu = np.sin(F(2 * np.pi) * c.U / p)
    wv = np.sin(F(2 * np.pi) * c.V / p)
    weave = 0.5 + 0.5 * wu * wv
    vs = c.V[c.mask > 0]
    lo, hi = (float(vs.min()), float(vs.max())) if len(vs) else (0.0, 1.0)
    f = np.clip((c.V - F(lo)) / F(max(hi - lo, 1e-3)), 0, 1)
    color = rgb(SAFFRON)[None, None] * (1 - f[..., None]) + rgb(SUNSET)[None, None] * f[..., None]
    sheen = noise(c, c.seed, 1.4 * c.scale, octaves=2, su=0.3, sv=3.0)
    color = color * (0.93 + 0.05 * weave[..., None] + 0.05 * sheen[..., None])
    height = 0.0005 * weave
    spec = 0.2 + 0.06 * weave + 0.05 * sheen
    c.rough = 0.5 - 0.08 * weave
    c.metal = full(c, 0.0)
    alpha = 0.46 + 0.1 * weave + 0.12 * f
    free = smoothstep(lo + 0.35, lo + 0.6, c.V)
    base = (color, height, spec, c.rough, c.metal)
    color2, height2, spec2 = gold_thread_band(c, color, height, spec, 0.08, 0.42, "chevron", 0.36)
    color = base[0] * (1 - free[..., None]) + color2 * free[..., None]
    height = base[1] * (1 - free) + height2 * free
    spec = base[2] * (1 - free) + spec2 * free
    c.rough = base[3] * (1 - free) + c.rough * free
    c.metal = base[4] * (1 - free) + c.metal * free
    border = np.clip(smoothstep(0.06, 0.1, c.dist) * smoothstep(0.44, 0.4, c.dist) +
                     smoothstep(0.05, 0.02, c.dist), 0, 1) * free
    c.alpha = np.clip(alpha * (1 - border) + border, 0, 1)
    return color, height, spec


def leather(c):
    grain = 0.6 * pebbles(c, 0.09 * c.scale, c.seed) + 0.4 * pebbles(c, 0.18 * c.scale, c.seed + 1)
    mott = noise(c, c.seed + 2, 1.6 * c.scale)
    warm = noise(c, c.seed + 3, 1.2 * c.scale)
    color = rgb((58, 38, 26))[None, None] * (1 + 0.09 * mott[..., None]) * (0.82 + 0.26 * grain[..., None])
    color = color + rgb((10, 5, 2))[None, None] * np.clip(warm, 0, None)[..., None]
    height = 0.007 * grain
    spec = (0.24 + 0.12 * grain) * (1 + 0.1 * mott)
    c.rough = 0.62 - 0.12 * grain
    c.metal = full(c, 0.0)
    return color, height, spec


GOLD = (212, 164, 74)


def gold_base(c, worn=0.0):
    mott = noise(c, c.seed, 1.4 * c.scale)
    brush = noise(c, c.seed + 1, 0.4 * c.scale, octaves=2, su=4.0, sv=0.1)
    color = rgb(GOLD)[None, None] * (1 + 0.06 * mott[..., None] + 0.03 * brush[..., None])
    patina = smoothstep(0.3, 1.0, noise(c, c.seed + 2, 0.8 * c.scale)) * worn
    color = color * (1 - 0.35 * patina[..., None]) + rgb((92, 74, 40))[None, None] * 0.35 * patina[..., None]
    height = 0.0012 * brush
    spec = 0.75 * (1 - 0.4 * patina)
    c.rough = 0.24 + 0.06 * brush + 0.25 * patina
    c.metal = full(c, 1.0)
    c.env = 0.9 * (1 - 0.4 * patina)
    return color, height, spec


def engrave(c, color, height, spec, groove, depth=0.006):
    g = np.clip(groove, 0, 1)
    color = color * (1 - 0.55 * g[..., None])
    height = height - depth * g
    spec = spec * (1 - 0.5 * g)
    c.rough = c.rough + 0.3 * g
    if c.env is not None:
        c.env = c.env * (1 - 0.5 * g)
    return color, height, spec


def inlay(c, color, height, spec, m, col, rough=0.12, spec_v=0.6):
    m = np.clip(m, 0, 1)
    color = color * (1 - m[..., None]) + rgb(col)[None, None] * m[..., None]
    height = height - 0.003 * m
    spec = spec * (1 - m) + spec_v * m
    c.rough = c.rough * (1 - m) + rough * m
    c.metal = c.metal * (1 - m)
    if c.env is not None:
        c.env = c.env * (1 - m) + 0.25 * m
    return color, height, spec


def sym_curve(ctrl, samples=16):
    """mk_fit.sym_curve (a curve over the angle round the body, mirrored left/right), without the Blender modules."""
    import mk_geom
    pts = np.array(ctrl, dtype=np.float64)
    ext = np.vstack([[-pts[1][0], pts[1][1]], pts, [2 * np.pi - pts[-2][0], pts[-2][1]]])
    dense = mk_geom.catmull_rom(ext, samples)
    dense = dense[(dense[:, 0] >= 0) & (dense[:, 0] <= np.pi)]
    order = np.argsort(dense[:, 0])
    a, z = dense[order, 0], dense[order, 1]

    def f(theta):
        x = np.abs(np.mod(np.asarray(theta) + np.pi, 2 * np.pi) - np.pi)
        return np.interp(x, a, z)
    return f


def torque(c, color, height, spec):
    """The Collar's torque (U = 6 x the angle round the neck, V = distance from the neck): fine flutes fanning out
    from the neck like a corona between two smooth polished edge bands, each band pierced with a row of tiny
    squares (dark recesses)."""
    import collar_dims as pc
    th = c.U.astype(np.float64) / 6.0
    lo = sym_curve(pc.INNER)(th.ravel()).reshape(th.shape).astype(F)
    hi = sym_curve(pc.OUTER)(th.ravel()).reshape(th.shape).astype(F)
    v = c.V
    edge = np.minimum(v - lo, hi - v)                   # distance to the nearer edge
    band = smoothstep(0.42, 0.34, edge)                 # the smooth edge bands
    # flutes: a rounded ridge every 0.32 in U (about 0.25 at the inner edge, wider outward, like rays)
    ph = fract(c.U / F(0.32))
    ridge = (0.5 + 0.5 * np.cos(F(2 * np.pi) * ph)) ** F(1.5)
    flute = (1 - band) * smoothstep(0.42, 0.55, edge)
    height = height + 0.007 * flute * ridge
    color, height, spec = engrave(c, color, height, spec, 0.45 * flute * smoothstep(0.35, 0.05, ridge), depth=0.002)
    # a fine engraved line where each band meets the flutes
    color, height, spec = engrave(c, color, height, spec, smoothstep(0.035, 0.0, np.abs(edge - 0.45)))
    # the piercings: squares 0.09 x 0.07 every 0.17 in U, centred in each band
    sq_u = np.abs(fract(c.U / F(0.17)) - 0.5) * F(0.17)
    sq_v = np.abs(edge - 0.2)
    hole = smoothstep(0.05, 0.04, sq_u) * smoothstep(0.04, 0.03, sq_v) * band
    color, height, spec = inlay(c, color, height, spec, hole, (24, 18, 12), rough=0.6, spec_v=0.05)
    height = height - 0.004 * hole
    # polished: less wear than the rest of the gold
    c.rough = c.rough * (1 - 0.45 * (1 - hole))
    return color, height, spec


def gold(c):
    k = c.key
    color, height, spec = gold_base(c, worn=0.15 if k != "collar" else 0.03)
    if k == "collar":
        color, height, spec = torque(c, color, height, spec)
    elif k in ("garter", "armletL", "armletR", "braceletL", "braceletR"):
        mid = 0.5 * (c.V.min() + c.V.max()) if (c.mask > 0).any() else 0.0
        vs = c.V[c.mask > 0]
        lo, hi = (vs.min(), vs.max()) if len(vs) else (0, 1)
        mid = 0.5 * (lo + hi)
        ridge = smoothstep(0.18, 0.0, np.abs(c.V - mid))
        height = height + 0.006 * ridge
        rays = np.abs(fract(c.U / F(0.6)) - 0.5) * 2
        chev = smoothstep(0.08, 0.0, np.abs(np.abs(c.V - mid) - (0.18 + 0.32 * rays)))
        color, height, spec = engrave(c, color, height, spec, chev * (np.abs(c.V - mid) > 0.2))
        color, height, spec = engrave(c, color, height, spec, smoothstep(0.05, 0.0, np.abs(c.V - lo - 0.12)) +
                                      smoothstep(0.05, 0.0, np.abs(c.V - hi + 0.12)))
    elif k == "coin":
        # a stamped sun coin (every coin shares this disc): raised rim, a ring of dots, twelve rays, a domed sun
        r = np.hypot(c.U, c.V) / F(COIN_UV)
        a = np.arctan2(c.V, c.U)
        rim = smoothstep(0.84, 0.88, r) * smoothstep(1.0, 0.95, r)
        dots = smoothstep(0.045, 0.02, np.hypot(np.abs(fract(a / F(2 * np.pi / 24)) - 0.5) * F(2 * np.pi / 24) *
                                              r, r - 0.74))
        ray_w = np.abs(fract(a / F(2 * np.pi / 12)) - 0.5) * 2
        rays = smoothstep(0.32, 0.36, r) * smoothstep(0.66, 0.6, r) * \
            smoothstep(0.2, 0.08, ray_w - (0.66 - r) * 0.9)
        core = smoothstep(0.3, 0.26, r)
        height = height + 0.006 * rim + 0.004 * dots + 0.004 * rays + 0.005 * core * (1 - (r / 0.3) ** 2)
        color, height, spec = engrave(c, color, height, spec, smoothstep(0.03, 0.0, np.abs(r - 0.82)) +
                                      smoothstep(0.03, 0.0, np.abs(r - 0.29)))
        # wear: the raised parts polished, the sunk field a little darker
        field = (1 - rim) * (1 - dots) * (1 - rays) * (1 - core) * smoothstep(0.9, 0.86, r)
        color = color * (1 - 0.18 * field[..., None])
        c.rough = c.rough + 0.12 * field
    elif k in ("medallion", "buckle", "circletsun", "backsun", "navelsun", "wristsun"):
        r = np.hypot(c.U, c.V)
        a = np.arctan2(c.V, c.U)
        spokes = smoothstep(0.1, 0.02, np.abs(fract(a / F(2 * np.pi / 24)) - 0.5) * 2 * r * F(np.pi / 12) /
                            np.maximum(r, 0.05)) * smoothstep(0.35, 0.45, r)
        circ = smoothstep(0.04, 0.0, np.abs(r - 0.42)) + smoothstep(0.04, 0.0, np.abs(r - 0.92))
        color, height, spec = engrave(c, color, height, spec, np.clip(circ + 0.6 * spokes, 0, 1))
    elif k.startswith("plate"):
        vs = c.V[c.mask > 0]
        us = c.U[c.mask > 0]
        if len(vs):
            cu, cv = 0.5 * (us.min() + us.max()), 0.5 * (vs.min() + vs.max())
            r = np.hypot(c.U - cu, c.V - cv)
            a = np.arctan2(c.V - cv, c.U - cu)
            sunm = smoothstep(0.05, 0.0, np.abs(r - 0.55)) + smoothstep(0.05, 0.0, np.abs(r - 0.25)) + \
                smoothstep(0.08, 0.0, np.abs(fract(a / F(2 * np.pi / 12)) - 0.5) * r) * smoothstep(0.6, 0.65, r) * \
                smoothstep(1.2, 1.0, r)
            color, height, spec = engrave(c, color, height, spec, np.clip(sunm, 0, 1))
        color, height, spec = engrave(c, color, height, spec, smoothstep(0.05, 0.0, np.abs(c.dist - 0.14)))
    elif k == "circlet":
        color, height, spec = engrave(c, color, height, spec, smoothstep(0.03, 0.0, np.abs(c.U)) * 0.8)
    return color, height, spec


def steel(c):
    import piece_scythe as ps
    x = c.U.astype(np.float64) / ps.UV_SCALE - ps.UV_X
    y = ps.UV_Y - c.V.astype(np.float64) / ps.UV_SCALE
    p = np.stack([x.ravel(), y.ravel()], 1)
    edge = ps.polyline_dist(p, ps.edge_line()).reshape(x.shape).astype(F)
    spine = ps.polyline_dist(p, np.array(ps.SPINE)).reshape(x.shape).astype(F)
    hammer = pebbles(c, 0.5 * c.scale * ps.UV_SCALE, c.seed)
    mott = noise(c, c.seed + 1, 1.2 * c.scale * ps.UV_SCALE)
    grind = noise(c, c.seed + 2, 0.25 * c.scale * ps.UV_SCALE, octaves=2, su=1.6, sv=0.08)
    bevel = smoothstep(ps.EDGE_BAND + 1.2, ps.EDGE_BAND + 0.05, edge)
    forged = rgb((30, 30, 34))[None, None] * (0.9 + 0.14 * hammer[..., None]) * (1 + 0.08 * mott[..., None])
    polished = rgb((92, 94, 102))[None, None] * (1 + 0.08 * grind[..., None])
    color = forged * (1 - bevel[..., None]) + polished * bevel[..., None]
    height = 0.004 * hammer * (1 - bevel) + 0.002 * grind * bevel
    spec = 0.3 + 0.05 * hammer + 0.45 * bevel
    c.rough = 0.42 - 0.08 * hammer - 0.24 * bevel
    c.metal = full(c, 1.0)
    c.env = 0.55 + 0.35 * bevel
    # gold filigree: a line along the spine with a wave of curls, and a sun near the root
    line = smoothstep(0.07, 0.03, np.abs(spine - 1.15))
    wave = np.sin(F(2 * np.pi) * x.astype(F) / F(3.2)) * F(0.45)
    curl = smoothstep(0.06, 0.02, np.abs(spine - (2.3 + wave))) * smoothstep(1.2, 2.5, edge)
    r = np.hypot(x - (-6.0), y - 47.8).astype(F)
    sun = smoothstep(0.06, 0.02, np.abs(r - 1.3)) + smoothstep(0.06, 0.02, np.abs(r - 0.7)) + \
        smoothstep(0.1, 0.03, np.abs(fract(np.arctan2(y - 47.8, x + 6.0).astype(F) / F(2 * np.pi / 12)) - 0.5) *
                   r) * smoothstep(1.35, 1.45, r) * smoothstep(2.2, 2.0, r)
    m = np.clip(line + curl + sun, 0, 1)
    gcol = rgb(GOLD)[None, None]
    color = color * (1 - m[..., None]) + gcol * m[..., None]
    height = height - 0.003 * m
    spec = spec * (1 - m) + 0.7 * m
    c.rough = c.rough * (1 - m) + 0.26 * m
    c.env = c.env * (1 - m) + 0.85 * m
    return color, height, spec


def ember(c):
    import piece_scythe as ps
    x = c.U.astype(np.float64) / ps.UV_SCALE - ps.UV_X
    y = ps.UV_Y - c.V.astype(np.float64) / ps.UV_SCALE
    p = np.stack([x.ravel(), y.ravel()], 1)
    edge = ps.polyline_dist(p, ps.edge_line()).reshape(x.shape).astype(F)
    s = np.clip(edge / ps.EDGE_BAND, 0, 1)
    flick = noise(c, c.seed, 0.35 * c.scale * ps.UV_SCALE, octaves=3)
    hot = (1 - s) ** 1.4 * (0.9 + 0.1 * flick)
    dark = rgb((40, 26, 18))[None, None]
    fire = rgb((246, 150, 46))[None, None]
    color = dark * (1 - hot[..., None]) + fire * hot[..., None]
    color = color + rgb((255, 222, 150))[None, None] * ((1 - s) ** 7)[..., None] * 0.5
    height = full(c, 0.0)
    spec = full(c, 0.6)
    c.rough = full(c, 0.3)
    c.metal = 0.6 * (1 - hot)
    c.env = 0.4 * (1 - hot)
    c.glow = (rgb((255, 168, 60))[None, None] * (hot ** 1.2)[..., None]).astype(F)
    return color, height, spec


def wood(c):
    grain = noise(c, c.seed, 0.6 * c.scale, octaves=3, su=0.08, sv=3.0)
    fine = noise(c, c.seed + 1, 0.15 * c.scale, octaves=2, su=0.05, sv=2.0)
    color = rgb((34, 22, 16))[None, None] * (1 + 0.18 * grain[..., None] + 0.08 * fine[..., None])
    height = 0.003 * grain + 0.001 * fine
    spec = 0.34 + 0.06 * grain
    c.rough = 0.38 - 0.06 * grain
    c.metal = full(c, 0.0)
    return color, height, spec


def obsidian(c):
    r = np.hypot(c.U, c.V)
    ripple = 0.5 + 0.5 * np.sin(r / F(0.22) + 2.0 * noise(c, c.seed, 1.2 * c.scale, octaves=2))
    color = rgb((14, 13, 18))[None, None] * (0.85 + 0.25 * ripple[..., None])
    rim = smoothstep(3.3, 3.85, r)
    color = color * (1 - rim[..., None]) + rgb((60, 30, 10))[None, None] * rim[..., None] * 0.6
    height = 0.0015 * ripple
    spec = full(c, 0.85)
    c.rough = 0.08 + 0.04 * ripple
    c.metal = full(c, 0.0)
    c.env = full(c, 0.7)
    return color, height, spec


MATERIALS = {"linen": linen, "leather": leather, "gold": gold, "steel": steel, "ember": ember, "wood": wood,
             "obsidian": obsidian, "gem": gold, "chiffon": chiffon}

# uniform patches: colour, specular, variation, roughness, metallic, environment
PATCHES = {
    "patch:lining": ((214, 206, 188), 0.05, 0.03, 0.88, 0.0, 0.0),
    "patch:edge": ((62, 40, 26), 0.28, 0.05, 0.6, 0.0, 0.0),
    "patch:strap": ((60, 40, 27), 0.3, 0.06, 0.58, 0.0, 0.0),
    "patch:leather": ((58, 38, 26), 0.3, 0.06, 0.6, 0.0, 0.0),
    "patch:gold": ((214, 166, 76), 0.8, 0.05, 0.24, 1.0, 0.9),
    "patch:goldworn": ((176, 134, 64), 0.6, 0.08, 0.4, 1.0, 0.7),
    "patch:amber": ((222, 118, 28), 0.85, 0.06, 0.08, 0.0, 0.35),
    "patch:garnet": ((128, 18, 22), 0.85, 0.06, 0.08, 0.0, 0.35),
    "patch:obsidian": ((14, 13, 18), 0.85, 0.04, 0.08, 0.0, 0.6),
    "patch:cord": ((150, 102, 52), 0.15, 0.05, 0.75, 0.0, 0.0),
    "patch:sole": ((48, 34, 24), 0.18, 0.06, 0.75, 0.0, 0.0),
    "patch:steel": ((52, 52, 58), 0.5, 0.06, 0.32, 1.0, 0.6),
    "patch:ebony": ((30, 21, 16), 0.3, 0.08, 0.4, 0.0, 0.0),
    "patch:ember": ((246, 150, 46), 0.4, 0.06, 0.3, 0.0, 0.0),
}
GLOW = {"patch:ember": (255, 168, 60)}


def stitch_rows(key, mat):
    if mat == "leather":
        return [0.14]
    return []


def stitches(mask, scale, rows, length=0.2, gap=0.12, width=0.045, hole=0.032):
    h, w = mask.shape
    thread = np.zeros((h, w), np.uint8)
    holes = np.zeros((h, w), np.uint8)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    tw = max(1, int(round(width * scale)))
    hr = max(1, int(round(hole * scale * 4)))
    for d in rows:
        iso = (dist >= d * scale).astype(np.uint8)
        contours, _ = cv2.findContours(iso, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            pts = cnt[:, 0, :].astype(np.float64)
            if len(pts) < 12:
                continue
            pts = np.vstack([pts, pts[:1]])
            acc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
            total = acc[-1]
            n = int(total / ((length + gap) * scale))
            if n < 2:
                continue
            pitch = total / n
            s = np.arange(n) * pitch
            a = np.stack([np.interp(s, acc, pts[:, k]) for k in range(2)], 1)
            b = np.stack([np.interp(s + length / (length + gap) * pitch, acc, pts[:, k]) for k in range(2)], 1)
            ia = np.round(a * 4).astype(np.int32)
            ib = np.round(b * 4).astype(np.int32)
            for p, q in zip(ia, ib):
                cv2.circle(holes, (int(p[0]), int(p[1])), hr, 255, -1, cv2.LINE_AA, shift=2)
                cv2.circle(holes, (int(q[0]), int(q[1])), hr, 255, -1, cv2.LINE_AA, shift=2)
            for p, q in zip(ia, ib):
                cv2.line(thread, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), 255, tw, cv2.LINE_AA, shift=2)
    return thread.astype(F) / 255, holes.astype(F) / 255


def normals(height, scale):
    hp = np.pad(height, 1, mode="edge")
    nx = (hp[1:-1, :-2] - hp[1:-1, 2:]) * F(scale * 0.5)
    ny = (hp[:-2, 1:-1] - hp[2:, 1:-1]) * F(scale * 0.5)
    inv = 1 / np.sqrt(nx * nx + ny * ny + 1)
    return np.stack([nx * inv, ny * inv, inv], -1)


def to_u8(x):
    return np.clip(np.round(x * 255), 0, 255).astype(np.uint8)


def rasterize_ids(parts, keys, size):
    ids = np.full((size, size), 255, np.uint8)
    index = {k: i for i, k in enumerate(keys)}
    for p in parts:
        if p["island"] not in index:
            continue
        val = index[p["island"]]
        for tri in np.round(p["uv"][p["t"]] * 4).astype(np.int32):
            cv2.fillConvexPoly(ids, tri, val, cv2.LINE_8, 2)
    return ids


def ao_map(parts, size):
    s = size // AO_FACTOR
    acc = np.zeros((s, s), F)
    cnt = np.zeros((s, s), F)
    for p in parts:
        ao = p["attrs"].get("ao")
        if ao is None:
            continue
        tris = np.round(p["uv"][p["t"]] / AO_FACTOR * 4).astype(np.int32)
        vals = ao[p["t"]].mean(1)
        tmp = np.zeros((s, s), F)
        hit = np.zeros((s, s), F)
        for tri, val in zip(tris, vals):
            cv2.fillConvexPoly(tmp, tri, float(val), cv2.LINE_8, 2)
            cv2.fillConvexPoly(hit, tri, 1.0, cv2.LINE_8, 2)
        acc += tmp * hit
        cnt += hit
    known = (cnt > 0).astype(F)
    acc = np.where(cnt > 0, acc / np.maximum(cnt, 1), 0).astype(F)
    wgt = cv2.GaussianBlur(known, (0, 0), 1.5)
    out = cv2.GaussianBlur(acc * known, (0, 0), 1.5) / np.maximum(wgt, 1e-4)
    out[wgt < 1e-3] = 1.0
    return np.clip(out, 0, 1)


def ao_region(ao, x0, y0, x1, y1):
    qx0, qy0 = x0 // AO_FACTOR, y0 // AO_FACTOR
    qx1, qy1 = -(-x1 // AO_FACTOR), -(-y1 // AO_FACTOR)
    q = ao[qy0:qy1, qx0:qx1]
    up = cv2.resize(q, ((qx1 - qx0) * AO_FACTOR, (qy1 - qy0) * AO_FACTOR), interpolation=cv2.INTER_LINEAR)
    return up[y0 - qy0 * AO_FACTOR:y1 - qy0 * AO_FACTOR, x0 - qx0 * AO_FACTOR:x1 - qx0 * AO_FACTOR]


def region(isl, size):
    return (max(0, isl["x"] - MARGIN), max(0, isl["y"] - MARGIN),
            min(size, isl["x"] + isl["w"] + MARGIN), min(size, isl["y"] + isl["h"] + MARGIN))


def paint_island(key, isl, mat, ids, idx, ao, size):
    x0, y0, x1, y1 = region(isl, size)
    h, w = y1 - y0, x1 - x0
    scale = isl["scale"]
    yy, xx = np.mgrid[0:h, 0:w].astype(F)
    lx, ly = xx + F(x0 - isl["x"]), yy + F(y0 - isl["y"])
    if isl["rot"]:
        lx, ly = F(isl["ext_u"]) - ly, lx
    U = F(isl["origin"][0]) + lx / F(scale)
    V = F(isl["origin"][1]) + ly / F(scale)
    mask = (ids[y0:y1, x0:x1] == idx).astype(np.uint8)
    a = ao_region(ao, x0, y0, x1, y1)
    if isl["rot"]:
        U, V, mask, a = (np.ascontiguousarray(np.rot90(x, -1)) for x in (U, V, mask, a))
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5) / F(scale)
    seed = 1009 + 7919 * (sum(map(ord, key)) % 997)
    c = Ctx(U, V, mask, dist, a, seed, scale, key)
    color, height, spec = MATERIALS.get(mat, leather)(c)
    rows = stitch_rows(key, mat)
    if rows:
        T, H = stitches(mask, scale, rows)
        height = height + 0.012 * np.sqrt(T) - 0.005 * H * (1 - T)
        color = color * (1 - 0.4 * H * (1 - T))[..., None]
        color = color * (1 - T[..., None]) + rgb(THREAD)[None, None] * (0.75 + 0.25 * a)[..., None] * T[..., None]
        spec = spec * (1 - T) + 0.3 * T
        c.rough = c.rough * (1 - T) + 0.5 * T
    if mat in ("leather", "linen"):
        groove = smoothstep(0.08, 0.0, dist)
        height = height - 0.01 * groove ** 2
        color = color * (1 - 0.25 * groove[..., None])
    albedo = color.copy()
    lo = 0.5 if mat == "linen" else 0.6
    color = color * (lo + (1 - lo) * a)[..., None]
    spec = spec * (0.45 + 0.55 * a)
    rough = np.clip(c.rough if c.rough is not None else full(c, 0.6), 0.03, 1)
    metal = np.clip(c.metal if c.metal is not None else full(c, 0.0), 0, 1)
    env = c.env if c.env is not None else full(c, 0.0)
    env = env * np.ones_like(spec)
    glow = c.glow * mask[..., None] if c.glow is not None else None
    alpha = c.alpha * np.ones_like(spec) if c.alpha is not None else None
    outs = [color, albedo, height, spec, rough, metal, a, env]
    if isl["rot"]:
        outs = [np.ascontiguousarray(np.rot90(x, 1)) for x in outs]
        if glow is not None:
            glow = np.ascontiguousarray(np.rot90(glow, 1))
        if alpha is not None:
            alpha = np.ascontiguousarray(np.rot90(alpha, 1))
    color, albedo, height, spec, rough, metal, a, env = outs
    nrm = normals(height, scale)
    return (x0, y0, x1, y1), to_u8(color), to_u8(albedo), to_u8(nrm * 0.5 + 0.5), to_u8(spec), to_u8(rough), \
        to_u8(metal), to_u8(a), to_u8(env), glow, (to_u8(alpha) if alpha is not None else None)


def paint_patch(key, isl, size, ao):
    x0, y0, x1, y1 = region(isl, size)
    h, w = y1 - y0, x1 - x0
    col, spec_v, var, rough_v, metal_v, env_v = PATCHES.get(key, ((128, 128, 128), 0.2, 0.05, 0.6, 0.0, 0.0))
    seed = 31 + sum(map(ord, key))
    yy, xx = np.mgrid[0:h, 0:w].astype(F)
    c = Ctx(xx, yy, np.ones((h, w), np.uint8), None, None, seed, 1.0, key)
    low = noise(c, seed, 48.0)
    fine = noise(c, seed + 1, 6.0, octaves=2)
    color = rgb(col)[None, None] * (1 + var * low[..., None] + 0.4 * var * fine[..., None])
    spec = np.full((h, w), spec_v, F) * (1 + 0.15 * low)
    rough = np.clip(np.full((h, w), rough_v, F) * (1 + 0.1 * low), 0.03, 1)
    nrm = np.zeros((h, w, 3), F)
    nrm[..., 2] = 1
    return (x0, y0, x1, y1), to_u8(color), to_u8(color), to_u8(nrm * 0.5 + 0.5), to_u8(spec), to_u8(rough), \
        np.full((h, w), int(metal_v * 255), np.uint8), np.full((h, w), 230, np.uint8), \
        np.full((h, w), int(env_v * 255), np.uint8)


def write_bgra_dds(path, levels):
    h, w = levels[0].shape[:2]
    header = bytearray(128)
    struct.pack_into("<4sIIIIIII", header, 0, b"DDS ", 124, 0x2100F, h, w, w * 4, 0, len(levels))
    struct.pack_into("<IIIIIIII", header, 76, 32, 0x41, 0, 32, 0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000)
    struct.pack_into("<I", header, 108, 0x401008)
    with open(path, "wb") as fh:
        fh.write(header)
        for lvl in levels:
            fh.write(np.ascontiguousarray(lvl[..., [2, 1, 0, 3]]).tobytes())


def read_bgra_dds(path):
    raw = np.fromfile(path, np.uint8)
    h, w = struct.unpack_from("<II", raw, 12)
    count = struct.unpack_from("<I", raw, 28)[0]
    levels, pos = [], 128
    for i in range(count):
        hh, ww = max(h >> i, 1), max(w >> i, 1)
        levels.append(raw[pos:pos + hh * ww * 4].reshape(hh, ww, 4)[..., [2, 1, 0, 3]])
        pos += hh * ww * 4
    return levels


def normal_levels(nrm, alpha):
    """Mip chain for a normal map: averaged then renormalised (keeps the detail from turning flat or noisy)."""
    levels = [np.dstack([nrm, alpha])]
    cur = levels[0]
    while cur.shape[0] > 1:
        hh, ww = cur.shape[0] // 2, cur.shape[1] // 2
        nxt = np.empty((hh, ww, 4), np.uint8)
        for r in range(0, hh, 512):
            r1 = min(hh, r + 512)
            blk = cur[2 * r:2 * r1].astype(F) / 255
            blk = blk.reshape(r1 - r, 2, ww, 2, 4).mean(axis=(1, 3))
            n = blk[..., :3] * 2 - 1
            n /= np.maximum(np.linalg.norm(n, axis=2, keepdims=True), 1e-6)
            nxt[r:r1, :, :3] = to_u8(n * 0.5 + 0.5)
            nxt[r:r1, :, 3] = to_u8(blk[..., 3])
        levels.append(nxt)
        cur = nxt
    return levels


def texconv(src, out_dir, fmt="BC7_UNORM", name=None):
    args = [TEXCONV, "-nologo", "-y", "-f", fmt, "-m", "0", "-o", out_dir]
    if fmt.startswith("BC7"):
        args += ["-bc", "q"] if False else []
    subprocess.run(args + [src], check=True, stdout=subprocess.DEVNULL)
    if name:
        produced = os.path.join(out_dir, os.path.splitext(os.path.basename(src))[0] + ".dds")
        target = os.path.join(out_dir, name)
        if os.path.abspath(produced) != os.path.abspath(target):
            os.replace(produced, target)


def save_png(path, img):
    cv2.imwrite(path, img[..., ::-1] if img.ndim == 3 and img.shape[2] == 3 else
                (img[..., [2, 1, 0, 3]] if img.ndim == 3 else img), [cv2.IMWRITE_PNG_COMPRESSION, 1])


def export_sets(work_dir, out_dir, stem, has_glow):
    """sources in work_dir (8192 px) -> out_dir/<label>/{vanilla, pbr} BC7 sets; small maps (glow, environment)
    are a quarter of the atlas, the RMAOS map half."""
    for label, k in SETS:
        src = os.path.join(work_dir, label)
        os.makedirs(src, exist_ok=True)
        van = os.path.join(out_dir, label, "vanilla")
        pbr = os.path.join(out_dir, label, "pbr")
        os.makedirs(van, exist_ok=True)
        os.makedirs(pbr, exist_ok=True)
        jobs = [("d.png", van, stem + ".dds", 0), ("n.dds", van, stem + "_n.dds", 0), ("m.png", van, stem + "_m.dds", 2),
                ("albedo.png", pbr, stem + ".dds", 0), ("pn.dds", pbr, stem + "_n.dds", 0),
                ("rmaos.png", pbr, stem + "_rmaos.dds", 1)]
        if has_glow:
            jobs += [("g.png", van, stem + "_g.dds", 2), ("g.png", pbr, stem + "_g.dds", 2)]
        for suffix, dst, name, extra in jobs:
            path = os.path.join(work_dir, stem + "_" + suffix)
            tmp = os.path.join(src, stem + "_" + suffix)
            if suffix.endswith(".dds"):
                write_bgra_dds(tmp, read_bgra_dds(path)[k:])
            else:
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                size = img.shape[0] >> k
                if suffix in ("m.png", "g.png"):
                    size = (img.shape[0] * 4) >> (k + extra)   # stored at a quarter already
                if suffix == "rmaos.png":
                    size = img.shape[0] >> (k + 1)
                size = max(size, 256)
                if size != img.shape[0]:
                    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
                cv2.imwrite(tmp, img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            # the environment mask has no alpha (BC1): PG Patcher must not mistake it for a complex material
            texconv(tmp, dst, "BC1_UNORM" if suffix == "m.png" else "BC7_UNORM", name)
        log("%s %s set compressed" % (stem, label))


def paint(manifest, work_dir, stem):
    size = manifest["size"]
    layout = manifest["layout"]
    parts = manifest["parts"]
    keys = [k for k, v in layout["islands"].items() if not v["patch"]]
    mats = {}
    for p in parts:
        mats.setdefault(p["island"], p["mat"])
    ids = rasterize_ids(parts, keys, size)
    ao = ao_map(parts, size)
    log("%s: islands rasterized, ambient occlusion" % stem)
    color = np.zeros((size, size, 3), np.uint8)
    albedo = np.zeros((size, size, 3), np.uint8)
    nrm = np.full((size, size, 3), (128, 128, 255), np.uint8)
    spec = np.zeros((size, size), np.uint8)
    q = size // AO_FACTOR
    rmaos = np.zeros((size, size, 4), np.uint8)
    rmaos[..., 0] = 160
    rmaos[..., 2] = 255
    rmaos[..., 3] = 255
    glow = np.zeros((q, q, 3), np.uint8)
    env = np.zeros((size, size), np.uint8)
    # opacity in the diffuse/albedo alpha: opaque everywhere but sheer fabric (it only matters on shapes with an
    # alpha property, i.e. the chiffon veils)
    alpha = np.full((size, size), 255, np.uint8)
    has_glow = False
    for idx, key in enumerate(keys):
        (x0, y0, x1, y1), c, al, n, s, r, m, a, e, gl, op = paint_island(key, layout["islands"][key], mats[key], ids,
                                                                        idx, ao, size)
        sel = ids[y0:y1, x0:x1] == idx
        grow = cv2.dilate(sel.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        if op is not None:
            alpha[y0:y1, x0:x1][grow] = op[grow]
        for arr, val in ((color, c), (albedo, al), (nrm, n)):
            arr[y0:y1, x0:x1][grow] = val[grow]
        spec[y0:y1, x0:x1][grow] = s[grow]
        rmaos[y0:y1, x0:x1, 0][grow] = r[grow]
        rmaos[y0:y1, x0:x1, 1][grow] = m[grow]
        rmaos[y0:y1, x0:x1, 2][grow] = a[grow]
        env[y0:y1, x0:x1][grow] = e[grow]
        if gl is not None:
            has_glow = True
            g = (slice(y0 // AO_FACTOR, -(-y1 // AO_FACTOR)), slice(x0 // AO_FACTOR, -(-x1 // AO_FACTOR)))
            small = cv2.resize(gl, (g[1].stop - g[1].start, g[0].stop - g[0].start), interpolation=cv2.INTER_AREA)
            glow[g] = np.maximum(glow[g], to_u8(small))
        log("%s (%s)" % (key, mats[key]))
    for key, isl in layout["islands"].items():
        if not isl["patch"]:
            continue
        (x0, y0, x1, y1), c, al, n, s, r, m, a, e = paint_patch(key, isl, size, ao)
        color[y0:y1, x0:x1], albedo[y0:y1, x0:x1], nrm[y0:y1, x0:x1], spec[y0:y1, x0:x1] = c, al, n, s
        rmaos[y0:y1, x0:x1, 0], rmaos[y0:y1, x0:x1, 1], rmaos[y0:y1, x0:x1, 2] = r, m, a
        env[y0:y1, x0:x1] = e
        if key in GLOW:
            has_glow = True
            g = (slice(y0 // AO_FACTOR, -(-y1 // AO_FACTOR)), slice(x0 // AO_FACTOR, -(-x1 // AO_FACTOR)))
            glow[g] = GLOW[key]
    del ids
    log("%s: patches" % stem)
    os.makedirs(work_dir, exist_ok=True)
    save_png(os.path.join(work_dir, stem + "_preview.png"), cv2.resize(color, (2048, 2048),
                                                                        interpolation=cv2.INTER_AREA))
    save_png(os.path.join(work_dir, stem + "_d.png"), np.dstack([color, alpha]))
    del color
    save_png(os.path.join(work_dir, stem + "_albedo.png"), np.dstack([albedo, alpha]))
    del albedo, alpha
    write_bgra_dds(os.path.join(work_dir, stem + "_n.dds"), normal_levels(nrm, spec))
    write_bgra_dds(os.path.join(work_dir, stem + "_pn.dds"), normal_levels(nrm, np.full_like(spec, 255)))
    del nrm, spec
    save_png(os.path.join(work_dir, stem + "_rmaos.png"), rmaos)
    del rmaos
    envq = cv2.resize(env, (q, q), interpolation=cv2.INTER_AREA)
    save_png(os.path.join(work_dir, stem + "_m.png"), np.dstack([envq] * 3))
    save_png(os.path.join(work_dir, stem + "_g.png"), glow)
    log("%s: sources written" % stem)
    return has_glow


def main(work, out_dir, which=("outfit", "scythe"), sets=None, export_only=False):
    """sets: labels to export (default all); export_only: reuse the 8192 px sources in <work>\\tex (no repaint),
    e.g. to make the 2K and 4K sets later from the same painting."""
    global SETS
    if sets:
        SETS = tuple(s for s in SETS if s[0] in sets)
    sys.path.insert(0, os.path.join(TOOLS, "pieces"))
    sys.path.insert(0, os.path.join(TOOLS, "lib"))
    tex_work = os.path.join(work, "tex")
    for kind in which:
        mpath = os.path.join(work, "texture_manifest.pkl" if kind == "outfit" else "texture_manifest_scythe.pkl")
        manifest = pickle.load(open(mpath, "rb"))
        stem = manifest.get("stem", "SunDancer" if kind == "outfit" else "EclipseScythe")
        has_glow = kind == "scythe" if export_only else paint(manifest, tex_work, stem)
        export_sets(tex_work, out_dir, stem, has_glow and kind == "scythe")


if __name__ == "__main__":
    args = sys.argv[3:]
    sets = None
    if "--sets" in args:
        i = args.index("--sets")
        sets = args[i + 1].split(",")
        args = args[:i] + args[i + 2:]
    export_only = "--export-only" in args
    args = [a for a in args if a != "--export-only"]
    main(sys.argv[1], sys.argv[2], tuple(args) or ("outfit", "scythe"), sets, export_only)
