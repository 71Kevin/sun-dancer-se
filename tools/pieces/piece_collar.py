"""Sun Dancer Collar (slot 45): a polished gold torque round the base of the neck, a crescent broad under the
throat that tapers to a slim band at the nape. Its face is fluted with fine rays fanning out from the neck like a
corona, between polished rolled edges, each edge with a row of tiny square piercings. Under the throat sits an
eclipse: an obsidian disc in a raised bezel with one bright bead on its rim (the "diamond ring" of a total eclipse).
At the nape the torque closes with a small eclipse clasp; nothing hangs down the back.

The look follows modern sculptural collars (a crescent torque, a fluted collar, a pierced choker) in an original
form; geometry and textures are made here (the fluting and the piercings are in the "collar" texture island).
The torque lies on the body like the rest of the outfit (weights and BodySlide data from the skin under it, kept
off the skin under every slider), with heavily smoothed weights and no breast bones, so it moves as one piece with
the neck and shoulders."""
import numpy as np

import mk_blend
import mk_fit
import mk_geom
from mk_parts import Part, bind, solid_parts
from piece_wrap import dome, torus

from collar_dims import INNER, OUTER  # noqa: E402  (the torque's outline, shared with the texture painter)

OFF = 0.58          # the outer face off the skin; the lining is THICK under it (clears the halter straps and their
THICK = 0.22        # knot at the nape, about 0.33 off the skin)
INTIMATE_SKIP = ("Breast", "Nipple")


def geodesic_from(v, t, seeds, iters=400):
    e = mk_geom.unique_edges(t)
    ln = np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1)
    i = np.concatenate([e[:, 0], e[:, 1]])
    j = np.concatenate([e[:, 1], e[:, 0]])
    ll = np.concatenate([ln, ln])
    d = np.full(len(v), np.inf)
    d[seeds] = 0.0
    for _ in range(iters):
        nd = d.copy()
        np.minimum.at(nd, i, d[j] + ll)
        if np.array_equal(nd, d):
            break
        d = nd
    return d


NU = 240            # columns round the neck
NV = 14             # rows across the torque


def edge_curve(sv, st, f, th, cols, smooth=8):
    """A closed edge of the torque: where f crosses 0 on the body (piece_wrap.iso_line), averaged in `cols` bins of
    the angle round the neck, gaps filled, smoothed round the neck. Returns cols x 3 points (bin centres)."""
    from piece_wrap import iso_line
    pts, (ang,) = iso_line(sv, st, f, [th])
    k = np.floor((ang + np.pi) / (2 * np.pi) * cols).astype(int) % cols
    out = np.full((cols, 3), np.nan)
    for b in range(cols):
        sel = k == b
        if sel.any():
            out[b] = pts[sel].mean(0)
    ok = ~np.isnan(out[:, 0])
    idx = np.arange(cols)
    for d in range(3):
        out[:, d] = np.interp(idx, idx[ok], out[ok, d], period=cols)
    # left and right alike (bin k mirrors bin cols-1-k across x = 0): a torque is symmetric
    out = 0.5 * (out + out[::-1] * np.array([-1.0, 1.0, 1.0]))
    for _ in range(smooth):
        out = 0.5 * out + 0.25 * (np.roll(out, 1, 0) + np.roll(out, -1, 0))
    return out


def collar_surface(proxy, inner=INNER, outer=OUTER, off=OFF):
    """The torque's outer face as a regular grid (NV rows from the inner to the outer edge, NU + 1 columns round the
    neck with the seam at the nape, under the clasp): its edges are curves of constant distance from the neck
    opening, the rows between them lie on the body, all OFF off the skin, then smoothed into one rigid sweep of
    metal that bridges the hollows (throat notch, collarbones)."""
    v, t = proxy.v, proxy.t
    sel = (v[t][:, :, 2].min(1) > 96.0)
    sv, st, _, used = mk_geom.compact(v, t[sel])
    loops = mk_geom.boundary_loops(st)
    neck = max(loops, key=lambda lp: sv[lp][:, 2].mean())
    dist = geodesic_from(sv, st, neck)
    c = sv[neck].mean(0)
    th = np.arctan2(sv[:, 0] - c[0], sv[:, 1] - c[1])
    lo_f, hi_f = mk_fit.sym_curve(inner), mk_fit.sym_curve(outer)
    # the inner edge smoothed harder: on the body it dips into the throat notch
    a = edge_curve(sv, st, dist - lo_f(th), th, NU, smooth=60)
    b = edge_curve(sv, st, dist - hi_f(th), th, NU)
    ths = -np.pi + (np.arange(NU) + 0.5) * 2 * np.pi / NU
    f = np.linspace(0, 1, NV)[:, None, None]
    G = a[None] * (1 - f) + b[None] * f
    loc, nrm, _, _ = mk_blend.nearest(proxy.tree, G.reshape(-1, 3))
    G = (loc + nrm * off).reshape(NV, NU, 3)
    mirror = np.array([-1.0, 1.0, 1.0])
    for _ in range(30):
        # round the neck (the grid wraps) and across the torque (the edges only move along it): the profile
        # straightens over the hollows; push_out keeps it off the skin where the body bulges
        G = 0.5 * G + 0.25 * (np.roll(G, 1, axis=1) + np.roll(G, -1, axis=1))
        G[1:-1] = 0.5 * G[1:-1] + 0.25 * (G[:-2] + G[2:])
        G = mk_fit.push_out(G.reshape(-1, 3), proxy, off, iters=1).reshape(NV, NU, 3)
        G = 0.5 * (G + G[:, ::-1] * mirror)
    # the seam at the nape (under the clasp): one more column, a copy of the first, so the uv runs on to +pi
    G = np.concatenate([G, G[:, :1]], axis=1)
    ths = np.append(ths, ths[0] + 2 * np.pi)
    tri = mk_geom.grid_tris(NV, NU + 1)
    p = G.reshape(-1, 3)
    # faces outward
    fn = mk_geom.face_normals(p, tri)
    lp, _, _, _ = mk_blend.nearest(proxy.tree, p[tri].mean(1))
    if ((p[tri].mean(1) - lp) * fn).sum() < 0:
        tri = tri[:, ::-1].copy()
    d = (lo_f(ths)[None] * (1 - f[..., 0]) + hi_f(ths)[None] * f[..., 0])
    attrs = {"d": d.ravel(), "th": np.repeat(ths[None], NV, 0).ravel()}
    return p, tri, attrs, c


def no_breast(bones):
    return np.array([0.0 if any(k in b for k in INTIMATE_SKIP) else 1.0 for b in bones])


def face_point(v, t, th0, frac, a):
    """A point of the torque's outer face at angle th0 round the neck, `frac` of the way from the inner to the
    outer edge, and the face's normal there."""
    lo = mk_fit.sym_curve(INNER)(np.array([th0]))[0]
    hi = mk_fit.sym_curve(OUTER)(np.array([th0]))[0]
    target = lo + frac * (hi - lo)
    dth = np.abs(np.angle(np.exp(1j * (a["th"] - th0))))
    score = dth * 4.0 + np.abs(a["d"] - target)
    i = int(np.argmin(score))
    n = mk_geom.vertex_normals(v, t)[i]
    return v[i], mk_geom.normalize(n), i


def eclipse(center, n, up, radius, name, wD, bead=True):
    """An eclipse: obsidian disc in a raised gold bezel, a bright bead on the bezel (upper right). wD: the weights
    and slider data of the torque under it (it moves exactly with the torque)."""
    up = mk_geom.normalize(up - n * np.dot(up, n))
    side = np.cross(n, up)
    parts = []
    gold = []
    dv, dt, duv = dome(center + n * 0.06, n, radius, radius * 0.24, seg=32, rings=5)
    parts.append((name + "_disc", dv, dt, duv, "patch:obsidian", "gem"))
    rv, rt, ruv = torus(center + n * 0.08, n, up, radius * 1.1, radius * 0.11, 40, 8)
    gold.append((name + "_bezel", rv, rt, ruv, "patch:gold", "gold"))
    bv, bt, buv = dome(center + n * 0.02, n, radius * 1.32, 0.05, seg=36, rings=3)
    gold.append((name + "_seat", bv, bt, buv, "patch:gold", "gold"))
    if bead:
        p = center + n * 0.12 + (side * np.cos(0.8) + up * np.sin(0.8)) * radius * 1.1
        gv, gt, guv = dome(p, n, radius * 0.22, radius * 0.2, seg=14, rings=4)
        parts.append((name + "_bead", gv, gt, guv, "patch:amber", "gem"))
    out = []
    w, D = wD
    for nm, v, t, uv, isl, mat in parts + gold:
        out.append(Part(nm, "CollarGold", v, t, uv, isl, mat, np.repeat(w, len(v), 0), np.repeat(D, len(v), 0)))
    return out


def build(body, proxy, log=print):
    v, t, a, c = collar_surface(proxy)
    gb = no_breast(proxy.bones)
    w, D = bind(v, t, proxy, smooth_iters=24, bone_mask=gb)
    D = mk_fit.garment_conform(v, D, proxy)
    # U runs round the neck (the fluting is a function of U: rays fanning out from the neck), V across the torque
    uv = np.stack([a["th"] * 6.0, a["d"]], 1)
    parts = solid_parts("collar", "CollarGold", v, t, uv, w, D, THICK, ("collar", "patch:goldworn", "patch:gold"),
                        ("gold", "gold", "gold"), rim_segments=4)
    # (solid_parts: the outer face is the surface itself, the lining THICK under it)
    p0, n0, i0 = face_point(v, t, 0.0, 0.42, a)
    parts += eclipse(p0 + n0 * 0.02, n0, np.array([0, 0, 1.0]), 0.6, "throat", (w[i0:i0 + 1], D[i0:i0 + 1]))
    pb, nb, ib = face_point(v, t, np.pi, 0.5, a)
    parts += eclipse(pb + nb * 0.02, nb, np.array([0, 0, 1.0]), 0.24, "nape", (w[ib:ib + 1], D[ib:ib + 1]),
                     bead=False)
    log("collar: torque %d verts, %d parts" % (len(v), len(parts)))
    for p in parts:
        p.bones = list(proxy.bones)
        p.sliders = list(proxy.slider_names)
        p.shape = "CollarGold"
    return {"parts": parts}
