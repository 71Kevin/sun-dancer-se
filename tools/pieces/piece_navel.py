"""Sun Dancer Navel Charm (slot 49): a curved gold bar through the navel with an amber gem on top and, below, an
eclipse charm (a small gold sun round an obsidian disc) with a short chain and a sun coin. Everything is rigid on
the navel: one set of bone weights, and the slider data of the navel (it moves with the belly sliders)."""
import numpy as np

import mk_fit
import mk_geom
from ev_shapes import chain_links, coin, sun_disc, teardrop, torus
from mk_parts import Part
from piece_wrap import dome, landmarks


def sphere(center, radius, seg=12, rings=8):
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


def build(body, proxy, log=print):
    _, navel = landmarks(body.v, body)
    # the navel's rim: the skin around the deepest point, and the outward direction of the belly there
    import mk_blend
    loc, nrm, _, _ = mk_blend.nearest(proxy.tree, navel[None])
    n = mk_geom.normalize(nrm[0] + np.array([0, 1.0, 0]))
    up = mk_geom.normalize(np.array([0, 0, 1.0]) - n * n[2])
    side = np.cross(n, up)
    # on the skin around the navel (the proxy has the navel filled in, so these sit on its rim)
    rim_top = mk_fit.push_out((navel + up * 0.75)[None], proxy, 0.2)[0]
    rim_bot = mk_fit.push_out((navel - up * 0.7)[None], proxy, 0.15)[0]
    pieces = []
    gv, gt, guv = sphere(rim_top + n * 0.05, 0.17)
    pieces.append(("navel_gem", gv, gt, guv, "patch:amber", "gem"))
    # the curved bar between the gem and the charm, just visible above and below the navel
    for k, (a, b) in enumerate(((rim_top, rim_top - n * 0.35 - up * 0.15), (rim_bot - n * 0.35 + up * 0.15, rim_bot))):
        path = mk_geom.catmull_rom(np.array([a, 0.5 * (a + b) + n * 0.08, b]), 6)
        bv, bt, buv, _ = mk_geom.sweep(path, np.full(len(path), 0.05), 8, up=tuple(side))
        pieces.append(("navel_bar%d" % k, bv, bt, buv, "patch:gold", "gold"))
    bv, bt, buv = sphere(rim_bot + n * 0.04, 0.12)
    pieces.append(("navel_ball", bv, bt, buv, "patch:gold", "gold"))
    # the eclipse charm hanging from the lower ball
    jr, jt, juv = torus(rim_bot - up * 0.2 + n * 0.06, side, up, 0.11, 0.03, 12, 5)
    pieces.append(("navel_jump", jr, jt, juv, "patch:gold", "gold"))
    cc = mk_fit.push_out((rim_bot - up * 0.78)[None], proxy, 0.16)[0]
    sv, st, suv = sun_disc(cc, n, up, 0.36, 0.12, rays=12, ray_len=0.18, rings=3)
    pieces.append(("navel_sun", sv, st, suv, "navelsun", "gold"))
    ov, ot, ouv = dome(cc + n * 0.08, n, 0.21, 0.06, seg=20, rings=3)
    pieces.append(("navel_eclipse", ov, ot, ouv, "patch:obsidian", "gem"))
    rv, rt, ruv = torus(cc + n * 0.1, n, up, 0.23, 0.025, 24, 5)
    pieces.append(("navel_ring", rv, rt, ruv, "patch:gold", "gold"))
    top = cc - up * 0.62 + n * 0.08
    lv, lt, luv, _ = chain_links([top, top - up * 0.55], link=0.2, width=0.12, wire=0.025)
    pieces.append(("navel_chain", lv, lt, luv, "patch:gold", "gold"))
    dv, dt, duv = teardrop(top - up * 0.95, n, -up, 0.15, 0.55, seg=12, rings=8)
    pieces.append(("navel_drop", dv, dt, duv, "patch:amber", "gem"))
    for k, sgn in enumerate((-1, 1)):
        c0 = cc + side * sgn * 0.62 - up * 0.32 + n * 0.08
        jv, jt2, juv2 = torus(c0 + up * 0.25, n, up, 0.07, 0.02, 10, 4)
        pieces.append(("navel_coinring%d" % k, jv, jt2, juv2, "patch:gold", "gold"))
        cv, ct, cuv = coin(c0, n, up, radius=0.2, thick=0.04)
        pieces.append(("navel_coin%d" % k, cv, ct, cuv, "coin", "gold"))
    # rigid on the navel: the weights and slider data of the skin at the navel
    w, D, _ = mk_fit.transfer(navel[None], proxy)
    parts = []
    for name, v, t, uv, island, mat in pieces:
        p = Part(name, "NavelGold", v, t, uv, island, mat, np.repeat(w, len(v), 0), np.repeat(D, len(v), 0))
        p.bones = list(proxy.bones)
        p.sliders = list(proxy.slider_names)
        parts.append(p)
    log("navel charm: %d parts at %s" % (len(parts), np.round(navel, 2).tolist()))
    return {"parts": parts}
