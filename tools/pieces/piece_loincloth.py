"""Sun Dancer Loincloth (slot 52), the dancer's hip dress: linen panels to mid-thigh in front and behind.

A low leather hip belt with gold discs, an eclipse buckle (a gold sun around an obsidian disc) with three amber
drops, a fringe of small sun coins along its lower edge and gold chains swagged over both hips, from the side discs
to the back discs; the small of the back is plain belt over the back panel (round 5: the eclipse clasp there and the
two short sacrum chains that only held it were taken off). The panels hang from under the belt and swing with SMP
physics (FSMP): bone chains start where the cloth leaves the body, an invisible coarse copy of the cloth collides
with a decimated copy of the body (both private to the outfit), coins along the hems swing with the cloth. The panel
weights keep at most 4 bones per vertex and fade a bone out over a few rings where it has to go (mk_fit.
limit_influences): cut vertex by vertex at export, they creased the back panel in game. The panels follow the
pelvis (and the buttocks' bones) but not the thighs: the cloth rests on the body, and dragged by the thighs it bent
into an S in the hip-shot idles (round 6). On the left thigh a gold
garter with coins, on the right a chain garter with a sun pendant, both hung from the belt by chains down the outer
thigh.

Two items are made from it: the Loincloth with the briefs (build_with_briefs, the original look) and the Loincloth
without them (build); the briefs alone are piece_briefs. The loincloth is built once per run and shared."""
import os
import pickle

import numpy as np

import mk_blend
import mk_export
import mk_fit
import mk_geom
import mk_smp
from ev_shapes import (axial_rig, band, catenary, chain_links, coin, hanging_panel, panel_bind, rigid_bind,
                       stud, sun_disc, teardrop, torus)
from mk_parts import Part, solid_parts
from piece_wrap import dome, garment_bones, surface_path

BELT_CENTER = [(0, 70.9), (0.6, 71.6), (1.2, 72.8), (1.6, 73.3), (2.2, 73.2), (2.7, 72.9), (np.pi, 72.7)]
BELT_HALF = 0.68
BELT_OFF = 0.36
BELT_THICK = 0.14
PANEL_THICK = 0.08
DISCS = [0.62, 1.15, 1.7, 2.3]
GARTER = (17.0, 18.1)   # distance along the thigh axis from the hip joint
VARIANTS = {
    "short": dict(stem="Loincloth", prefix="EV Loin ",
                  front=dict(x_top=(-5.5, 5.5), x_bot=(-4.7, 4.7), z_bot=46.0, drop=1.2, off_top=0.26, off_bot=1.1,
                             chains=(0.18, 0.5, 0.82), segs=3, hem_coins=5, rows_step=0.2, uv_scale=1.0),
                  back=dict(x_top=(-6.0, 6.0), x_bot=(-5.2, 5.2), z_bot=47.5, drop=1.0, off_top=0.26, off_bot=1.0,
                            chains=(0.18, 0.5, 0.82), segs=3, hem_coins=5, rows_step=0.2, uv_scale=1.0),
                  body_z=(28.0, 84.0), physics=dict(mass=0.35, stiffness=14.0, swing=0.6)),
}


def skeleton():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return pickle.load(open(os.path.join(root, "cache", "ref", "skeleton.pkl"), "rb"))


def skin_weights(points, proxy):
    """Weights (and slider data) of the skin under each point: small rigid things follow the body right below."""
    w, D, _ = mk_fit.transfer(np.atleast_2d(points), proxy)
    return w, D


def rigid_part(name, shape, v, t, uv, island, mat, proxy, at):
    w, D = skin_weights(np.asarray(at)[None], proxy)
    return Part(name, shape, v, t, uv, island, mat, np.repeat(w, len(v), 0), np.repeat(D, len(v), 0))


# ---------------------------------------------------------------- belt and its jewellery
def outward(v, t, zs, cy):
    """Vertex normals of a band round the torso, turned away from the torso's axis."""
    n = mk_geom.vertex_normals(v, t)
    axis = np.stack([np.zeros(len(v)), np.interp(v[:, 2], zs, cy), v[:, 2]], 1)
    return n * np.sign(((v - axis) * n).sum(1))[:, None]


def belt_parts(proxy, stem, layer=None):
    center = mk_fit.sym_curve(BELT_CENTER)
    zs = np.arange(68.6, 75.2, 0.12)
    cy = mk_fit.torso_axis(proxy, zs)
    cols = 220
    p, t, theta, zz = mk_fit.ring_grid(proxy, zs, cols, cy)
    p = mk_fit.drape(p, t, proxy, BELT_OFF, iters=40, lam=0.45)
    p = mk_fit.push_out(mk_fit.fill_concave(p, t, np.ones(len(p)), iters=60), proxy, BELT_OFF, iters=2)
    # (the belt keeps its shape over the briefs' waistband: pushed out where the briefs came close, it rippled and its
    # stitching stretched; the briefs leave it the room instead, piece_briefs.cleft_mask)
    rows = len(zs)
    g = p.reshape(rows, cols, 3)
    ring = np.linalg.norm(np.diff(np.concatenate([g, g[:, :1]], 1), axis=1), axis=2)
    S = np.concatenate([np.zeros((rows, 1)), np.cumsum(ring, 1)[:, :-1]], 1).ravel()
    th = np.arctan2(np.sin(theta), np.cos(theta))
    # |theta|: the belt's height curve is symmetric, and |theta| varies continuously all round. Theta itself jumps
    # from +pi to -pi at the back; the cut vertices on the edges across the jump took a theta near 0 (the front's
    # height) and their texture rows jumped: the stitching broke at the middle of the back (hidden by the old clasp)
    a = {"S": S, "z0": zz, "theta": np.abs(th)}
    p, t, a = mk_geom.open_ring_seam(p, t, np.arange(rows) * cols, a, "S", np.cumsum(ring, 1)[:, -1])
    f = BELT_HALF - np.abs(a["z0"] - center(a["theta"]))
    v, tt, a = mk_geom.iso_cut(p, t, f, a)
    v = mk_geom.smooth_boundary(v, tt, iters=8, lam=0.4)
    from mk_parts import bind
    w, D = bind(v, tt, proxy, smooth_iters=4)
    D = mk_fit.garment_conform(v, D, proxy)
    if layer is not None:
        D = mk_fit.garment_conform(v, D, layer, slack_margin=0.5, normals=outward(v, tt, zs, cy), smooth_tris=tt,
                                   smooth_iters=60, min_cos=0.65)
    uv = np.stack([a["S"], a["z0"] - center(a["theta"]) + BELT_HALF], 1)
    from ev_shapes import island_splitter
    L = a["S"].max()
    parts = solid_parts("belt", stem, v, tt, uv, w, D, BELT_THICK, ("belt", "patch:lining", "patch:edge"),
                        ("leather", "lining", "edge"), rim_segments=2, attrs={"S": a["S"]},
                        split_outer=island_splitter("S", [L / 3, 2 * L / 3], "belt"))
    tree = mk_blend.bvh(v, tt)
    gold = stem + "Gold"

    def on_belt(th0, dz=0.0):
        """Point of the belt's outer face at angle th0 (0 = front), dz above or below its centre line; near the
        edges (smoothed after the cut) the ray may pass the belt, then it moves toward the centre line."""
        dvec = np.array([np.sin(th0), np.cos(th0), 0.0])
        for f in (1.0, 0.75, 0.5, 0.0):
            z = float(center(np.array([th0]))[0]) + dz * f
            o = np.array([0.0, float(np.interp(z, zs, cy)), z])
            loc, nrm, _, dist = mk_blend.raycast(tree, (o + dvec * 30)[None], -dvec[None], 40.0)
            if np.isfinite(dist[0]):
                n = nrm[0] if np.dot(nrm[0], dvec) > 0 else -nrm[0]
                return loc[0], n
        loc, nrm, _, _ = mk_blend.nearest(tree, (o + dvec * 14)[None])
        return loc[0], nrm[0] if np.dot(nrm[0], dvec) > 0 else -nrm[0]

    for k, th0 in enumerate([x for d in DISCS for x in (d, -d)]):
        c, n = on_belt(th0)
        sv, st, suv = stud(c + n * 0.04, n, [0, 0, 1.0], 0.48, 0.16, kind="disc")
        parts.append(rigid_part("belt_disc%d" % k, gold, sv, st, suv, "patch:gold", "gold", proxy, c))
    # the small of the back is left plain: the hip chains end at the back discs. (Round 4 had a small eclipse clasp
    # there with two short sacrum chains swagged to it from the back discs; before that a broad sun with a garnet and
    # three coins. In game both read as an ornament on the buttocks; round 5 took the clasp and its chains off.)
    # the eclipse buckle: a gold sun, an obsidian disc in its middle, a thin ring round the disc, three amber drops
    c, n = on_belt(0.0)
    c = c + n * 0.1
    bv, bt, buv = sun_disc(c, n, [0, 0, 1.0], 1.05, 0.2, rays=12, ray_len=0.45, rings=4)
    parts.append(rigid_part("buckle", gold, bv, bt, buv, "buckle", "gold", proxy, c))
    ov, ot, ouv = dome(c + n * 0.16, n, 0.55, 0.12, seg=24, rings=4)
    parts.append(rigid_part("buckle_eclipse", gold, ov, ot, ouv, "patch:obsidian", "gem", proxy, c))
    rv, rt, ruv = torus(c + n * 0.2, n, [0, 0, 1.0], 0.6, 0.05, 32, 6)
    parts.append(rigid_part("buckle_ring", gold, rv, rt, ruv, "patch:gold", "gold", proxy, c))
    down = mk_geom.normalize(np.array([0, 0, -1.0]) - n * n[2])
    side = np.cross(n, down)
    for k, (dx, length) in enumerate(((-0.55, 1.0), (0.0, 1.6), (0.55, 1.0))):
        top = c + side * dx - down * 0.0 + down * 0.85 + n * 0.12
        lv, lt, luv, _ = chain_links([top, top + down * length], link=0.24, width=0.14, wire=0.03)
        parts.append(rigid_part("buckle_chain%d" % k, gold, lv, lt, luv, "patch:gold", "gold", proxy, c))
        gc = top + down * (length + 0.45)
        gv, gt, guv = teardrop(gc, n, down, 0.24, 0.75, seg=12, rings=8)
        parts.append(rigid_part("buckle_drop%d" % k, gold, gv, gt, guv, "patch:amber", "gem", proxy, c))
    # sun coins along the lower edge (front and sides; the back panel hangs at the back)
    ths = np.concatenate([np.arange(0.3, 2.05, 0.085), -np.arange(0.3, 2.05, 0.085)])
    for k, th0 in enumerate(ths):
        e, n = on_belt(th0, -BELT_HALF + 0.08)
        down = mk_geom.normalize(np.array([0, 0, -1.0]) - n * n[2])
        jr, jt, juv = torus(e + down * 0.06 + n * 0.05, np.cross(n, down), down, 0.09, 0.025, 10, 4)
        cc = e + down * 0.44 + n * 0.08
        cv, ct, cuv = coin(cc, n, -down, radius=0.32, thick=0.05)
        parts.append(rigid_part("beltcoin%d" % k, gold, cv, ct, cuv, "coin", "gold", proxy, cc - n * 0.3))
        parts.append(rigid_part("beltcoinring%d" % k, gold, jr, jt, juv, "patch:gold", "gold", proxy, cc - n * 0.3))
    # gold chains swagged over both hips, from the side disc to the back disc
    for sgn in (-1, 1):
        a0, _ = on_belt(sgn * DISCS[1], -BELT_HALF + 0.1)
        a1, _ = on_belt(sgn * DISCS[3], -BELT_HALF + 0.1)
        for k, sag in enumerate((2.0, 3.4)):
            path = catenary(a0, a1, sag, 40)
            path = mk_fit.push_out(path, proxy, BELT_OFF + 0.18, iters=3)
            lv, lt, luv, al = chain_links(path, link=0.26, width=0.15, wire=0.03)
            w, D = skin_weights(lv, proxy)
            parts.append(Part("hipchain%d_%d" % (sgn, k), gold, lv, lt, luv, "patch:gold", "gold", w, D))
    return parts, center, on_belt


# ---------------------------------------------------------------- panels with physics
def grid_fill(G, free, iters=500):
    """Values of the free cells of a grid (rows x cols x F) replaced by the average of their neighbours, repeated:
    a harmonic fill between the fixed cells (Neumann at the grid's edges)."""
    G = np.array(G, dtype=np.float64)
    cnt = np.zeros(G.shape[:2])
    cnt[1:] += 1
    cnt[:-1] += 1
    cnt[:, 1:] += 1
    cnt[:, :-1] += 1
    for _ in range(iters):
        acc = np.zeros_like(G)
        acc[1:] += G[:-1]
        acc[:-1] += G[1:]
        acc[:, 1:] += G[:, :-1]
        acc[:, :-1] += G[:, 1:]
        G[free] = (acc / cnt[..., None])[free]
    return G


PANEL_CLEAR = 0.2     # the panels' least distance to the briefs below the belt


def briefs_layer(body, proxy):
    """The briefs' outer surface (welded), with their slider data: the panels keep clear of it."""
    import piece_briefs
    outer = [p for p in piece_briefs.parts(body, proxy) if p.shape == "Briefs"]
    off = np.cumsum([0] + [len(p.v) for p in outer[:-1]])
    v = np.concatenate([p.v for p in outer])
    t = np.concatenate([p.t + o for p, o in zip(outer, off)])
    D = np.concatenate([p.D for p in outer])
    wv, wt, first, _ = mk_fit.welded(v, t)
    return mk_fit.layer_body(wv, wt, D[first])




def keep_clear(pts, t, layer, normals, clear, weight=None):
    """The surface pushed out along `normals` (the push spread smoothly) where it comes closer than `clear` to the
    briefs. Where two cloth layers nearly touch, the inner one shows through in game, and moving with the body they
    cross."""
    pairs = mk_geom.neighbor_pairs(t, len(pts))
    for _ in range(3):
        sd, _, _, _ = layer.signed_distance(pts)
        need = np.clip(clear - sd, 0, None) * (np.abs(sd) < 1.5)
        if weight is not None:
            need = need * weight
        if need.max() < 1e-3:
            break
        g = need.copy()
        for _ in range(40):
            g = np.maximum(need, g + 0.5 * mk_geom.laplacian(g, pairs))
        pts = pts + normals * g[:, None]
    return pts


def panel_clear(pts, t, layer, side, center):
    """A panel kept PANEL_CLEAR off the briefs below the belt (under it all the layers share the belt's thickness)."""
    pn = mk_geom.vertex_normals(pts, t)
    pn = pn * np.sign(pn @ np.array([0, side * 1.0, 0]))[:, None]
    th = np.arctan2(pts[:, 0], pts[:, 1])
    shown = np.clip((center(th) - BELT_HALF - pts[:, 2]) / 0.6, 0, 1)
    return keep_clear(pts, t, layer, pn, PANEL_CLEAR, shown * shown * (3 - 2 * shown))


def panel_parts(proxy, body, center, side, cfg, name, stem, prefix, layer=None):
    z_top = float(center(np.array([0.0 if side > 0 else np.pi]))[0]) - BELT_HALF + 0.25
    pts, t, contact, X, Z, rows, cols = hanging_panel(proxy, side, cfg["x_top"], cfg["x_bot"], z_top, cfg["z_bot"],
                                                      drop=cfg["drop"], off_top=cfg["off_top"],
                                                      off_bot=cfg["off_bot"], rows_step=cfg["rows_step"])
    if layer is not None:
        pts = panel_clear(pts, t, layer, side, center)
    gb = garment_bones(proxy.bones)
    Wb, D = panel_bind(pts, t, proxy, body, contact, rows, cols, side, bone_mask=gb)
    # the cloth hangs from the belt and rests on the buttocks (or the belly) without being sewn to the skin: the
    # thighs' share of the skin's weights (up to a fifth at the panel's edges) goes to the pelvis. Following the
    # thighs, the part lying on the body was dragged up to 1 unit sideways when the hips turned against the legs (the
    # player's hip-shot idles turn them 15 to 21 degrees), while the chains below stay with the pelvis (FSMP's
    # springs hold them at rest): the panel bent into an S just under the buttocks (round 6). In the poses tested
    # the skin and the briefs stay under the panel without the thighs (tools/dev/panel_drape.py, docs/QA.md)
    thighs = np.array(["Thigh" in b for b in proxy.bones])
    pelvis_b = list(proxy.bones).index("NPC Pelvis [Pelv]")
    Wb[:, pelvis_b] += Wb[:, thighs].sum(1)
    Wb[:, thighs] = 0.0
    # across the middle the panel bridges the buttock cleft (and the gap between the thighs): there the skin's weights
    # switch from one side's buttock and thigh to the other's; blended into their left/right mean toward x = 0
    Wb = mk_fit.blend_midline(Wb, proxy.bones, pts[:, 0], 1.5)
    # the hanging cloth's slider data: a smooth (harmonic) fill from the rows lying on the body. Each column used to
    # copy the data of the row where it leaves the body; those rows lie at different heights (belly, mons), so
    # under the body sliders neighbouring columns moved apart into steps
    # only the top stretch lying on the belly or the buttocks is fixed: lower contacts (the fronts of the thighs
    # under a long panel) are filled too, the thighs' sideways sliders would pinch the cloth there
    C = contact.reshape(rows, cols)
    top = np.cumprod(C, axis=0).astype(bool)
    D = grid_fill(D.reshape(rows, cols, -1), ~top).reshape(D.shape).astype(np.float32)
    # keep the cloth in front of the body under every slider: pushed along the panel's own normal (forward or back;
    # pushed along the inner thighs' normals the two halves of the panel moved sideways into each other) and
    # spread smoothly over the panel (the nearest body point jumps between thigh, crotch and the other thigh)
    pn = mk_geom.vertex_normals(pts, t)
    pn = pn * np.sign((pn @ np.array([0, side * 1.0, 0])))[:, None]
    D = mk_fit.garment_conform(pts, D, proxy, slack_margin=0.6, normals=pn, smooth_tris=t, smooth_iters=120,
                               along_normals=True)
    if layer is not None:
        # and out at least as far as the briefs under it (worn together in one item)
        D = mk_fit.garment_conform(pts, D, layer, slack_margin=0.5, normals=pn, smooth_tris=t, smooth_iters=120,
                                   along_normals=True)
    grid = pts.reshape(rows, cols, 3)
    chains, chain_w = mk_smp.grid_rig(grid, contact.reshape(rows, cols), np.linspace(0, 1, cols), cfg["chains"],
                                      cfg["segs"], prefix + ("F" if side > 0 else "B"),
                                      np.array([0, side * 1.0, 0]))
    rr, cc = np.divmod(np.arange(rows * cols), cols)
    K, Wc = chain_w(rr.astype(float), cc.astype(float))
    bones = list(proxy.bones) + [n for ch in chains for n in ch.names]
    W = np.concatenate([Wb * (1 - K)[:, None], Wc * K[:, None]], 1)
    # the chain anchors ("... 00") are kinematic children of the pelvis (FSMP moves them with it): skinning to an
    # anchor is skinning to the pelvis, so their weight goes to the pelvis, two fewer bones where the cloth leaves the
    # body. Then at most 4 bones per vertex, each bone that has to go fading out over a few rings: where the skin
    # weights (pelvis, buttock, thigh) met two chains, up to 8 bones were cut vertex by vertex at export (jumps of 0.3
    # across one edge) and the back panel creased along a staircase of edges when the chains swung
    pelvis = bones.index("NPC Pelvis [Pelv]")
    for k, b in enumerate(bones):
        if b.endswith(" 00"):
            W[:, pelvis] += W[:, k]
            W[:, k] = 0.0
    W, rep = mk_fit.limit_influences(W, t)
    jumps = mk_fit.edge_jumps(W, t)
    print("    %s weights: cut %.3f -> %.3f in %d steps, largest jump across an edge %.3f"
          % (name, rep["cut_before"], rep["cut_after"], rep["steps"], jumps.max()), flush=True)
    L = np.concatenate([np.zeros(1), np.cumsum(np.linalg.norm(np.diff(grid[:, cols // 2], axis=0), axis=1))])
    uv = np.stack([X - X.min(), np.repeat(L, cols)], 1)
    if side < 0:
        uv[:, 0] = uv[:, 0].max() - uv[:, 0]
    uv = uv * cfg["uv_scale"]
    nrm = mk_geom.vertex_normals(pts, t)
    parts = solid_parts(name, stem, pts, t, uv, W, D, PANEL_THICK, (name, "patch:lining", "patch:edge"),
                        ("linen", "lining", "edge"), rim_segments=2, normals=nrm)
    for p in parts:
        p.bones = bones
    # coins along the hem swing with the cloth
    out = mk_geom.normalize(np.array([0, side * 1.0, 0]))
    gold = []
    for k, u in enumerate(np.linspace(0.08, 0.92, cfg["hem_coins"])):
        cf = u * (cols - 1)
        c0 = int(round(cf))
        r_hem = rows - 1
        e = grid[r_hem, c0] + out * (PANEL_THICK + 0.04)
        down = np.array([0, 0, -1.0])
        cv, ct, cuv = coin(e + down * 0.42, out, -down, radius=0.32, thick=0.05)
        jv, jt, juv = torus(e + down * 0.05, np.cross(out, down), down, 0.09, 0.025, 10, 4)
        kk, wc = chain_w(np.array([float(r_hem)]), np.array([cf]))
        wrow = np.concatenate([np.zeros(len(proxy.bones)), wc[0]])
        Dc = np.repeat(D[r_hem * cols + c0][None], 1, 0)
        for nm, vv, tt_, uu, isl in (("hemcoin", cv, ct, cuv, "coin"), ("hemring", jv, jt, juv, "patch:gold")):
            p = Part("%s%s%d" % (name, nm, k), stem + "Gold", vv, tt_, uu, isl, "gold",
                     np.repeat(wrow[None], len(vv), 0), np.repeat(Dc, len(vv), 0))
            p.bones = bones
            gold.append(p)
    # the invisible collider: the dynamic part of the cloth at a coarse resolution
    C = contact.reshape(rows, cols)
    first_free = []
    for u in cfg["chains"]:
        free = np.nonzero(~C[:, int(round(u * (cols - 1)))])[0]
        first_free.append(int(free[0]) if len(free) else rows - 2)
    r_lo = min(first_free)
    rows_at = np.linspace(r_lo + 0.35 * (rows - 1 - r_lo) / cfg["segs"], rows - 1, 3 * cfg["segs"])
    cols_at = np.linspace(0, cols - 1, 2 * len(cfg["chains"]) + 1)
    cv, ct, cW, (rf, cf) = mk_smp.coarse_collider(grid, chain_w, rows_at, cols_at)
    cv = cv - out * 0.05
    r0 = np.clip(np.round(rf).astype(int), 0, rows - 1)
    c0 = np.clip(np.round(cf).astype(int), 0, cols - 1)
    cD = D[r0 * cols + c0]
    col = Part(name + "_collider", "EV_PhysCloth", cv, ct, np.stack([cv[:, 0], cv[:, 2]], 1), "patch:edge",
               "collision", np.concatenate([np.zeros((len(cv), len(proxy.bones))), cW], 1), cD)
    col.bones = bones
    return parts + gold, chains, col


# ---------------------------------------------------------------- thighs: garter, chain garter, suspender chains
def thigh_axis(skel, s):
    return skel["NPC %s Thigh [%sThg]" % (s, s)][:3, 3], skel["NPC %s Calf [%sClf]" % (s, s)][:3, 3]


def garter_parts(proxy, skel, stem):
    gold = stem + "Gold"
    a, b = thigh_axis(skel, "L")
    rig = axial_rig(proxy, a, b, GARTER[0] - 3.0, GARTER[1] + 3.0, keep=("Thigh",), radius=8.0)
    parts, (v, tt, w, D, attrs, fr) = band(proxy, "garter", gold, a, b, GARTER[0], GARTER[1], 0.18, 0.12,
                                           "patch:gold", "gold", cols=120, step=0.08, radius=7.5, weights_fn=rig)
    for p in parts:
        p.shape = gold
        p.mat = "gold"
        p.island = "patch:goldworn" if p.name.endswith(("_in", "_rim")) else "garter"
    ax, u, wv = fr
    mid = a + ax * 0.5 * (GARTER[0] + GARTER[1])
    outdir = mk_geom.normalize(np.array([-1.0, 0.25, 0]) - ax * ax.dot([-1.0, 0.25, 0]))
    tree = mk_blend.bvh(v, tt)
    loc, nrm, _, _ = mk_blend.raycast(tree, (mid + outdir * 20)[None], -outdir[None], 30.0)
    n = nrm[0] if np.dot(nrm[0], outdir) > 0 else -nrm[0]
    sv, st, suv = sun_disc(loc[0] + n * 0.1, n, -ax, 0.62, 0.16, rays=10, ray_len=0.28, rings=3)
    rw = rig(loc[0][None])
    _, sD = rigid_bind(sv, proxy, loc[0])
    parts.append(Part("garter_sun", gold, sv, st, suv, "patch:gold", "gold", np.repeat(rw, len(sv), 0), sD))
    # three coins on short chains below the sun
    down = mk_geom.normalize(ax - n * np.dot(ax, n))
    side = np.cross(n, down)
    for k, (dx, length) in enumerate(((-0.75, 0.5), (0.0, 0.9), (0.75, 0.5))):
        top = loc[0] + n * 0.14 + side * dx + down * 0.75
        lv, lt, luv, _ = chain_links([top, top + down * length], link=0.24, width=0.14, wire=0.03)
        cc = top + down * (length + 0.36)
        cv, ct, cuv = coin(cc, n, -down, radius=0.3, thick=0.05)
        _, cD = rigid_bind(cv, proxy, cc - n * 0.4)
        for nm, vv, t_, uu, isl in (("garterchain", lv, lt, luv, "patch:gold"), ("gartercoin", cv, ct, cuv, "coin")):
            parts.append(Part("%s%d" % (nm, k), gold, vv, t_, uu, isl, "gold", np.repeat(rw, len(vv), 0),
                              np.repeat(cD[:1], len(vv), 0)))
    return parts


def chain_garter_parts(proxy, skel, stem):
    """Right thigh: a ring of fine chain at the garter's height, a small sun pendant with two coins outside."""
    gold = stem + "Gold"
    a, b = thigh_axis(skel, "R")
    ax = mk_geom.normalize(b - a)
    s = 0.5 * (GARTER[0] + GARTER[1])
    rig = axial_rig(proxy, a, b, s - 3.0, s + 3.0, keep=("Thigh",), radius=8.0)
    c = a + ax * s
    # rays from the middle of the thigh's cross-section (the joint axis runs off-centre)
    d = proxy.v - c
    h = d @ ax
    sel = (np.abs(h) < 0.4) & (np.linalg.norm(d - np.outer(h, ax), axis=1) < 8.0)
    c = c + (proxy.v[sel].mean(0) - c) - ax * np.dot(proxy.v[sel].mean(0) - c, ax)
    u = mk_geom.normalize(np.array([0, 1.0, 0]) - ax * ax[1])
    wv = np.cross(ax, u)
    ang = np.linspace(0, 2 * np.pi, 97)
    dirs = np.outer(np.cos(ang), u) + np.outer(np.sin(ang), wv)
    loc, nrm, _, dist = mk_blend.raycast(proxy.tree, np.repeat(c[None], len(ang), 0), dirs, 12.0)
    ring = loc + dirs * 0.16
    lv, lt, luv, _ = chain_links(ring, link=0.28, width=0.16, wire=0.032)
    parts = [Part("chaingarter", gold, lv, lt, luv, "patch:gold", "gold", rig(lv), skin_weights(lv, proxy)[1])]
    outdir = mk_geom.normalize(np.array([1.0, 0.25, 0]) - ax * ax.dot([1.0, 0.25, 0]))
    loc2, nrm2, _, _ = mk_blend.raycast(proxy.tree, (c + outdir * 20)[None], -outdir[None], 30.0)
    n = nrm2[0] if np.dot(nrm2[0], outdir) > 0 else -nrm2[0]
    top = loc2[0] + n * 0.2
    down = mk_geom.normalize(ax - n * np.dot(ax, n))
    side = np.cross(n, down)
    pv, pt, puv = sun_disc(top + down * 0.7 + n * 0.06, n, -down, 0.45, 0.14, rays=10, ray_len=0.22, rings=3)
    rw = rig(top[None])
    _, pD = rigid_bind(pv, proxy, top - n * 0.2)
    parts.append(Part("chaingarter_sun", gold, pv, pt, puv, "patch:gold", "gold", np.repeat(rw, len(pv), 0), pD))
    for k, dx in enumerate((-0.6, 0.6)):
        t0 = top + side * dx + down * 0.3
        clv, clt, cluv, _ = chain_links([t0, t0 + down * 0.8], link=0.24, width=0.14, wire=0.03)
        cc = t0 + down * 1.15
        cv, ct, cuv = coin(cc, n, -down, radius=0.28, thick=0.05)
        for nm, vv, t_, uu, isl in (("cgchain", clv, clt, cluv, "patch:gold"), ("cgcoin", cv, ct, cuv, "coin")):
            parts.append(Part("%s%d" % (nm, k), gold, vv, t_, uu, isl, "gold", np.repeat(rw, len(vv), 0),
                              np.repeat(pD[:1], len(vv), 0)))
    return parts


def suspender_parts(proxy, skel, on_belt, stem):
    """Chains from the side discs of the belt down the outer thighs to the garters."""
    gold = stem + "Gold"
    parts = []
    for s, sgn in (("L", -1), ("R", 1)):
        a, b = thigh_axis(skel, s)
        ax = mk_geom.normalize(b - a)
        start, n0 = on_belt(sgn * DISCS[1], -BELT_HALF + 0.05)
        outdir = mk_geom.normalize(np.array([sgn * 1.0, 0.15, 0]) - ax * ax.dot([sgn * 1.0, 0.15, 0]))
        mid = a + ax * (GARTER[0] - 0.3)
        loc, _, _, _ = mk_blend.raycast(proxy.tree, (mid + outdir * 20)[None], -outdir[None], 30.0)
        end = loc[0]
        ctrl = [start, 0.5 * (start + end) + outdir * 0.4, end]
        path, nrm = surface_path(proxy, ctrl, 0.26, samples=14, iters=3)
        lv, lt, luv, al = chain_links(path, link=0.28, width=0.16, wire=0.032)
        w, D = skin_weights(lv, proxy)
        parts.append(Part("suspender" + s, gold, lv, lt, luv, "patch:gold", "gold", w, D))
    return parts


# ---------------------------------------------------------------- physics body
def collision_body(proxy, zmin, zmax, ratio=0.06):
    """Decimated copy of the hips and legs (arms left out) for the cloth to collide with: weights collapsed to the
    few body bones the physics knows, slider data from the body (it follows BodySlide and OBody morphs)."""
    import bmesh
    import bpy
    from mk_parts import bind
    v, t = proxy.v, proxy.t
    keep = (v[t][:, :, 2].min(1) > zmin) & (v[t][:, :, 2].max(1) < zmax)
    arm = proxy.mask_bones(("UpperArm", "Forearm", "Hand", "Clavicle", "Finger"))
    keep &= arm[t].max(1) < 0.05
    sv, st, _, _ = mk_geom.compact(v, t[keep])
    ob = mk_blend.mesh_object("colproxy", sv, st, smooth=False)
    mod = ob.modifiers.new("dec", "DECIMATE")
    mod.ratio = ratio
    dg = bpy.context.evaluated_depsgraph_get()
    em = ob.evaluated_get(dg).to_mesh()
    bm = bmesh.new()
    bm.from_mesh(em)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.verts.index_update()
    cv = np.array([p.co[:] for p in bm.verts])
    ct = np.array([[p.index for p in f.verts] for f in bm.faces])
    bm.free()
    ob.evaluated_get(dg).to_mesh_clear()
    bpy.data.objects.remove(ob)
    w, D = bind(cv, ct, proxy)
    w, bones = mk_export.collapse_to_smp_bones(w, proxy.bones)
    p = Part("physbody", "EV_PhysBody", cv, ct, np.stack([cv[:, 0], cv[:, 2]], 1), "patch:edge", "collision", w, D)
    p.bones = bones
    return p


# ---------------------------------------------------------------- build
_BUILT = {}


def build_with_briefs(body, proxy, log=print):
    """The Loincloth as first made: the loincloth over the tie-side briefs (one item)."""
    import piece_briefs
    res = dict(build(body, proxy, log))
    res["parts"] = list(res["parts"]) + list(piece_briefs.parts(body, proxy))
    log("loincloth with the briefs: %d parts" % len(res["parts"]))
    return res


def build(body, proxy, log=print, skel=None, variant="short"):
    """The loincloth alone (belt, panels on physics, garters, chains): built once per run, shared by both items."""
    key = (id(proxy), variant)
    if key not in _BUILT:
        _BUILT[key] = _build(body, proxy, log, skel, variant)
    res = _BUILT[key]
    return dict(res, parts=list(res["parts"]))


def _build(body, proxy, log=print, skel=None, variant="short"):
    cfg = VARIANTS[variant]
    stem, prefix = cfg["stem"], cfg["prefix"]
    skel = skel if skel is not None else skeleton()
    layer = briefs_layer(body, proxy)
    parts, center, on_belt = belt_parts(proxy, stem, layer)
    log("%s: belt, buckle, %d coins, hip chains" % (stem, sum(p.name.startswith("beltcoin") and "ring" not in p.name
                                                             for p in parts)))
    chains, colliders = [], []
    for side, key in ((1, "front"), (-1, "back")):
        name = "panel_" + key
        pp, ch, col = panel_parts(proxy, body, center, side, cfg[key], name, stem, prefix, layer)
        parts += pp
        chains += ch
        colliders.append(col)
    log("%s: panels, %d chains of %d bones" % (stem, len(chains), len(chains[0].names)))
    parts += garter_parts(proxy, skel, stem)
    parts += chain_garter_parts(proxy, skel, stem)
    parts += suspender_parts(proxy, skel, on_belt, stem)
    log("%s: garters and suspender chains" % stem)
    pbody = collision_body(proxy, *cfg["body_z"])
    chain_bones = [n for ch in chains for n in ch.names]
    for p in parts:
        if getattr(p, "bones", None) is None or len(p.bones) != p.w.shape[1]:
            p.bones = list(proxy.bones)
        p.sliders = list(proxy.slider_names)
        if p.mat in ("gold", "gem"):
            # chains in a shape of their own (a BSTriShape holds at most 65535 triangles)
            p.shape = stem + ("Chains" if any(k in p.name for k in ("chain", "suspender")) else "Gold")
        elif p.name.endswith("_in"):
            p.shape = stem + "Lining"
        else:
            p.shape = stem
    # one collider shape for both panels
    cv = np.concatenate([c.v for c in colliders])
    off = np.cumsum([0] + [len(c.v) for c in colliders[:-1]])
    ct = np.concatenate([c.t + o for c, o in zip(colliders, off)])
    cbones = list(proxy.bones) + chain_bones
    cw = np.zeros((len(cv), len(cbones)))
    for c, o in zip(colliders, off):      # each panel's collider knows only its own chains: map by bone name
        for j, b in enumerate(c.bones):
            cw[o:o + len(c.v), cbones.index(b)] += c.w[:, j]
    cD = np.concatenate([c.D for c in colliders])
    collider = Part("collider", "EV_PhysCloth", cv, ct, np.stack([cv[:, 0], cv[:, 2]], 1), "patch:edge", "collision",
                    cw, cD)
    collider.bones = cbones
    collider.sliders = list(proxy.slider_names)
    pbody.sliders = list(proxy.slider_names)
    parts += [collider, pbody]
    # node hierarchy and bone globals for the export, the SMP system
    glob = {n: skel[n] for n in mk_smp.SKELETON_CHAIN}
    hierarchy = [(mk_smp.SKELETON_CHAIN[i], mk_smp.SKELETON_CHAIN[i - 1] if i else None)
                 for i in range(len(mk_smp.SKELETON_CHAIN))]
    for ch in chains:
        for n, m in zip(ch.names, ch.mats):
            glob[n] = m
        hierarchy += ch.hierarchy()
    kinematic = list(dict.fromkeys(pbody.bones))
    xml = mk_smp.system_xml(kinematic, chains, "EV_PhysBody", "EV_PhysCloth", **cfg["physics"])
    log("%s: %d parts, physics %d kinematic + %d chain bones" % (stem, len(parts), len(kinematic), len(chain_bones)))
    return {"parts": parts, "fx": {"globals": glob, "hierarchy": hierarchy, "xml": xml, "chains": chains}}
