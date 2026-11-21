"""Sun Dancer Arm Veils (slot 59): sheer saffron chiffon veils that fall from a fine gold cord just below the
armlets behind the outer side of each arm to the middle of the forearm, with an embroidered border and small sun
coins on the hem. A separate item (the veils can be worn or left off; with the Armlets they look sewn to them).

The veils swing with SMP physics (FSMP), like the hip panels: bone chains hang from the upper arm, the cloth blends
from the arm's own weights at the cord to the chains below it (at most 4 bones per vertex, fading out where one has to
go, the chain anchors counted as the upper arm they hang from: round 5); an invisible copy of the veils (per-vertex)
collides with a decimated copy of the arms and torso (per-triangle, the arms finer), both private to the veils.

Where the veil may go (tools/dev/veil_gravity.py hangs it under gravity in the player's idles, walks, runs, sneaking
and sitting): only the sector between the outer side and the back of the arm. Wrapped further toward the front, the
bent forearm went through it; further toward the body, it swung into the hips when walking and was pinched between
the arm and the ribs. The chains may swing far away from the arm but not toward it (asymmetric limits, see
mk_smp.system_xml), and only a little sideways: under gravity the flared veil used to fall in round the hanging arm,
which then showed through it. The rest shape is therefore what an idle shows. The pleats only stand outward from the
veil's inner surface, which keeps its distance to the arm. The veil itself is a single sheet with a double-sided,
alpha-blended material (the chiffon's sheerness is in the texture's alpha)."""
import os
import pickle

import numpy as np

import mk_blend
import mk_fit
import mk_geom
import mk_smp
from ev_shapes import coin, limb_grid
from mk_parts import Part

# the skeleton's chain from the root to the upper arms (XP32 Maximum Skeleton, with its CME nodes)
TORSO_CHAIN = ["NPC", "NPC Root [Root]", "NPC COM [COM ]", "CME Body [Body]", "CME UBody [UBody]",
               "NPC Spine [Spn0]", "CME Spine [Spn0]", "NPC Spine1 [Spn1]", "CME Spine1 [Spn1]", "NPC Spine2 [Spn2]",
               "CME Spine2 [Spn2]"]


def arm_chain(s):
    return ["CME %s Shoulder" % s, "NPC %s Clavicle [%sClv]" % (s, s), "CME %s Clavicle [%sClv]" % (s, s),
            "NPC %s UpperArm [%sUar]" % (s, s)]


CORD_S = 2.2        # the cord: this far below the middle of the upper arm (the armlets end 1.7 below it)
LENGTH = 16.5       # the veil's length along the upper arm's axis (to the middle of the forearm, arm hanging)
CENTER = 0.87       # the veil's centre line, this far round from the outer side of the arm toward its back (50 deg)
SPREAD = (0.70, 0.70)   # how far it reaches round the arm from there: toward the back, toward the outer side
FLARE = 3.0         # how far the hem stands off the arm
GAP = 0.32          # the veil's top under the cord, off the skin
ROWS_STEP = 0.3
COLS = 33
PLEATS = 2.5        # soft folds across the veil from the gathering at the cord, opening toward the hem
SIDE_SWING = 0.2    # chain limits (radians): sideways, in the cloth's plane
SWING_OUT = 1.0     # away from the arm
SWING_IN = 0.02     # toward the arm
CHAINS = (0.18, 0.5, 0.82)
SEGS = 5
HEM_COINS = 5
UV_SCALE = 0.55     # the chiffon is a fine, even fabric: painted at about half the outfit's texel density
SHAPE = "Veils"
GOLD = "VeilsGold"


def skeleton():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return pickle.load(open(os.path.join(root, "cache", "ref", "skeleton.pkl"), "rb"))


def arm_ring(proxy, a, e, s, cols=96):
    """The skin round the upper arm at distance s along it: points, normals, the ring's centre and the angles of the
    points in the arm's frame."""
    p, n, t, ss, th, fr, rows, cc = limb_grid(proxy, a, e, s - 0.1, s + 0.1, 0.1, cols, radius=4.5)
    mid = rows // 2
    P = p.reshape(rows, cc, 3)[mid]
    N = n.reshape(rows, cc, 3)[mid]
    return P, N, P.mean(0), th[mid * cc:(mid + 1) * cc], fr


def veil_parts(proxy, skel, side):
    s = "L" if side < 0 else "R"
    a = skel["NPC %s UpperArm [%sUar]" % (s, s)][:3, 3]
    e = skel["NPC %s Forearm [%sLar]" % (s, s)][:3, 3]
    Lu = float(np.linalg.norm(e - a))
    s_cord = 0.5 * Lu + CORD_S
    P, N, c, th, (ax, u, w) = arm_ring(proxy, a, e, s_cord)
    radial = P - c - np.outer((P - c) @ ax, ax)
    r_skin = np.linalg.norm(radial, axis=1)
    # the veil's centre line: the outer back of the arm (away from the body and behind it)
    want = np.array([side * np.cos(CENTER), -np.sin(CENTER), 0.0])
    want = mk_geom.normalize(want - ax * np.dot(want, ax))
    th_c = np.arctan2(np.dot(want, w), np.dot(want, u))
    # columns run from the body side (behind the arm) to the outer side (round toward the front); which way round
    # that is depends on the frame: a small turn that moves the direction away from the body's middle is outward
    t_lin = np.linspace(-1, 1, COLS)
    span = np.where(t_lin < 0, SPREAD[0], SPREAD[1])
    step = np.cos(th_c + 0.3) * u + np.sin(th_c + 0.3) * w
    turn = 1.0 if (step[0] - want[0]) * side > 0 else -1.0
    thetas = th_c + turn * t_lin * span
    dirs = np.outer(np.cos(thetas), u) + np.outer(np.sin(thetas), w)
    ang = np.arctan2(radial @ w, radial @ u)
    order = np.argsort(ang)
    rr = np.interp(np.mod(thetas + np.pi, 2 * np.pi) - np.pi, ang[order], r_skin[order], period=2 * np.pi)
    # hem: longest on the centre line, shorter toward the edges
    lengths = LENGTH * (0.86 + 0.14 * np.cos(0.5 * np.pi * t_lin))
    rows = int(round(LENGTH / ROWS_STEP)) + 1
    f = np.linspace(0, 1, rows)[:, None]
    S = f * lengths[None, :]
    # gathered at the cord, the chiffon opens at once (away from the arm, whose oval section turns under it with
    # the upper arm's twist bone) and then falls nearly straight
    R_base = rr[None, :] + GAP + FLARE * f ** 0.55
    # pleats standing outward from the veil's inner surface (R_base), so they never bring it nearer the arm
    folds = 0.5 + 0.5 * np.sin(2 * np.pi * PLEATS * np.arange(COLS) / (COLS - 1))[None, :]
    R = R_base + (0.06 + 0.42 * f ** 1.15) * folds
    # c is the arm's centre at the cord: the veil starts just under the cord and runs down the arm's axis
    grid = c[None, None] + ax[None, None] * (S - 0.1)[..., None] + R[..., None] * dirs[None]
    inner = c[None, None] + ax[None, None] * (S - 0.1)[..., None] + R_base[..., None] * dirs[None]
    inner = mk_fit.push_out(inner.reshape(-1, 3), proxy, 0.25, iters=2).reshape(rows, COLS, 3)
    pts = grid.reshape(-1, 3)
    t = mk_geom.grid_tris(rows, COLS)
    # faces outward (away from the arm)
    fn = mk_geom.face_normals(pts, t)
    if ((pts[t].mean(1) - c - np.outer((pts[t].mean(1) - c) @ ax, ax)) * fn).sum() < 0:
        t = t[:, ::-1].copy()
    pts = mk_fit.push_out(pts, proxy, 0.25, iters=2)
    grid = pts.reshape(rows, COLS, 3)
    # slider data at the cord (the skin's own, per column); the hanging cloth follows its column's top
    top_skin = c + dirs * rr[:, None]
    tw, tD, _ = mk_fit.transfer(top_skin, proxy)
    tD = mk_fit.garment_conform(grid[0], tD, proxy)
    D = np.repeat(tD[None], rows, 0).reshape(rows * COLS, -1, 3).astype(np.float32)
    # weights of the part still held by the arm (above the chains): the skin right under each point, so the veil
    # turns with the upper arm's twist bone like the skin does (the cord's weights on every row let the oval arm
    # turn through it when the arms swung)
    Wb, _, _ = mk_fit.transfer(pts, proxy)
    Wb = mk_geom.smooth(Wb, mk_geom.neighbor_pairs(t, len(pts)), iters=6, lam=0.5)
    Wb /= np.maximum(Wb.sum(1, keepdims=True), 1e-9)
    top = np.zeros((rows, COLS), dtype=bool)
    top[:2] = True
    pn = mk_geom.vertex_normals(pts, t)
    D = mk_fit.garment_conform(pts, D, proxy, slack_margin=0.6, normals=pn, smooth_tris=t, smooth_iters=60,
                               along_normals=True)
    # physics chains down the veil
    chains, chain_w = mk_smp.grid_rig(grid, top, np.linspace(0, 1, COLS), CHAINS, SEGS, "EV Veil %s" % s,
                                      want, parent="NPC %s UpperArm [%sUar]" % (s, s), blend_segs=0.3)
    rr_i, cc_i = np.divmod(np.arange(rows * COLS), COLS)
    K, Wc = chain_w(rr_i.astype(float), cc_i.astype(float))
    bones = list(proxy.bones) + [n for ch in chains for n in ch.names]
    W = np.concatenate([Wb * (1 - K)[:, None], Wc * K[:, None]], 1)
    # the chain anchors ("... 00") are kinematic children of the upper arm (FSMP moves them with it): their weight goes
    # to that bone; then at most 4 bones per vertex, each bone that has to go fading out (mk_fit.limit_influences).
    # Just below the cord the arm's bones (upper arm, twist) met two chains' anchors and first bones: cut vertex by
    # vertex at export, the weights jumped by 0.53 across one edge (a crease under the cord when the veil swung)
    arm = bones.index("NPC %s UpperArm [%sUar]" % (s, s))
    for k, b in enumerate(bones):
        if b.endswith(" 00"):
            W[:, arm] += W[:, k]
            W[:, k] = 0.0
    W, _ = mk_fit.limit_influences(W, t)
    # uv: true arc length across each row, distance down the veil
    arc = (thetas - thetas[0]) * np.sign(thetas[-1] - thetas[0])
    uv = np.stack([(arc[None, :] * R_base).ravel(), S.ravel()], 1) * UV_SCALE
    veil = Part("veil" + s, SHAPE, pts, t, uv, "veil", "chiffon", W, D)
    veil.normals = pn
    veil.bones = bones
    parts = [veil]
    # the gold cord round the arm
    ring = P + N * (GAP - 0.02)
    for _ in range(10):
        ring = 0.5 * ring + 0.25 * (np.roll(ring, 1, axis=0) + np.roll(ring, -1, axis=0))
    ring = mk_fit.push_out(ring, proxy, GAP - 0.05, iters=2)
    loop = np.vstack([ring, ring[:1]])
    cv, ct, cuv, cols = mk_geom.sweep(loop, np.full(len(loop), 0.075), 8, up=tuple(ax), closed=True)
    rw, rD, _ = mk_fit.transfer(P, proxy)
    rD = mk_fit.garment_conform(ring, rD, proxy)
    idx = np.repeat(np.concatenate([np.arange(len(ring)), [0]]), cols)[:len(cv)]
    cord = Part("veilcord" + s, GOLD, cv, ct, cuv, "patch:gold", "gold", rw[idx], rD[idx])
    cord.bones = list(proxy.bones)
    parts.append(cord)
    # small sun coins on the hem, swinging with the cloth
    for k, uu in enumerate(np.linspace(0.1, 0.9, HEM_COINS)):
        cf = uu * (COLS - 1)
        c0 = int(round(cf))
        hem = grid[-1, c0]
        out = dirs[c0]
        down = mk_geom.normalize(grid[-1, c0] - grid[-2, c0])
        cc_ = hem + down * 0.42 + out * 0.04
        kv, kt, kuv = coin(cc_, out, -down, radius=0.26, thick=0.045)
        _, wc = chain_w(np.array([float(rows - 1)]), np.array([cf]))
        wrow = np.concatenate([np.zeros(len(proxy.bones)), wc[0]])
        p = Part("veilcoin%s%d" % (s, k), GOLD, kv, kt, kuv, "coin", "gold", np.repeat(wrow[None], len(kv), 0),
                 np.repeat(D[(rows - 1) * COLS + c0][None], len(kv), 0))
        p.bones = bones
        parts.append(p)
    # the invisible collider: the dynamic part of the veil's inner surface (under the pleats), one sphere about
    # every 0.8 across and 1.0 down
    rel = 2
    rows_at = np.linspace(rel + 0.35 * (rows - 1 - rel) / SEGS, rows - 1, 16)
    cols_at = np.linspace(0, COLS - 1, 9)
    kv, kt, kW, (rf, cfr) = mk_smp.coarse_collider(inner, chain_w, rows_at, cols_at)
    r0 = np.clip(np.round(rf).astype(int), 0, rows - 1)
    c0 = np.clip(np.round(cfr).astype(int), 0, COLS - 1)
    col = Part("veilcollider" + s, "EV_PhysCloth", kv, kt, np.stack([kv[:, 0], kv[:, 2]], 1), "patch:edge",
               "collision", np.concatenate([np.zeros((len(kv), len(proxy.bones))), kW], 1), D[r0 * COLS + c0])
    col.bones = bones
    return parts, chains, col


SMP_ARM_BONES = ("NPC Pelvis [Pelv]", "NPC Spine [Spn0]", "NPC Spine1 [Spn1]", "NPC Spine2 [Spn2]",
                 "NPC L UpperArm [LUar]", "NPC R UpperArm [RUar]", "NPC L Forearm [LLar]", "NPC R Forearm [RLar]")


def smp_arm_bone(name):
    for s in ("L", "R"):
        if "NPC %s Upperarm" % s in name or "NPC %s UpperArm" % s in name:
            return "NPC %s UpperArm [%sUar]" % (s, s)
        if "NPC %s Forearm" % s in name or "NPC %s Hand" % s in name:
            return "NPC %s Forearm [%sLar]" % (s, s)
    if any(k in name for k in ("Spine2", "Clavicle", "Breast", "Neck")):
        return "NPC Spine2 [Spn2]"
    if any(k in name for k in ("Spine1", "Belly")):
        return "NPC Spine1 [Spn1]"
    if "Spine [" in name:
        return "NPC Spine [Spn0]"
    return "NPC Pelvis [Pelv]"


def decimated(v, t, ratio, name):
    import bmesh
    import bpy
    sv, st, _, _ = mk_geom.compact(v, t)
    ob = mk_blend.mesh_object(name, sv, st, smooth=False)
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
    return cv, ct


def collision_body(proxy, zmin=64.0, zmax=113.0, ratio=0.05, arm_ratio=0.22):
    """Decimated copy of the arms and torso for the veils to collide with: weights collapsed to the few bones the
    physics knows (each arm keeps its upper arm and forearm), slider data from the body. The arms, which the veils
    hang along, are kept about four times finer than the torso (a coarse arm lies well inside the skin)."""
    from mk_parts import bind
    v, t = proxy.v, proxy.t
    keep = (v[t][:, :, 2].min(1) > zmin) & (v[t][:, :, 2].max(1) < zmax)
    arm = proxy.mask_bones(("UpperArm", "Upperarm", "Forearm", "Hand"))
    on_arm = (arm[t] > 0.5).all(1)
    pieces = [decimated(v, t[keep & on_arm], arm_ratio, "veilcolarms"),
              decimated(v, t[keep & ~on_arm], ratio, "veilcoltorso")]
    cv = np.concatenate([p[0] for p in pieces])
    ct = np.concatenate([pieces[0][1], pieces[1][1] + len(pieces[0][0])])
    w, D = bind(cv, ct, proxy)
    out = list(SMP_ARM_BONES)
    wn = np.zeros((len(w), len(out)))
    for j, b in enumerate(proxy.bones):
        wn[:, out.index(smp_arm_bone(b))] += w[:, j]
    wn /= np.maximum(wn.sum(1, keepdims=True), 1e-9)
    p = Part("physbody", "EV_PhysBody", cv, ct, np.stack([cv[:, 0], cv[:, 2]], 1), "patch:edge", "collision", wn, D)
    p.bones = out
    return p


def build(body, proxy, log=print, skel=None):
    skel = skel if skel is not None else skeleton()
    parts, chains, colliders = [], [], []
    for side in (-1, 1):
        pp, ch, col = veil_parts(proxy, skel, side)
        parts += pp
        chains += ch
        colliders.append(col)
    log("veils: 2 sheets, %d chains of %d bones" % (len(chains), len(chains[0].names)))
    pbody = collision_body(proxy)
    chain_bones = [n for ch in chains for n in ch.names]
    for p in parts:
        if getattr(p, "bones", None) is None or len(p.bones) != p.w.shape[1]:
            p.bones = list(proxy.bones)
        p.sliders = list(proxy.slider_names)
        p.shape = GOLD if p.mat in ("gold", "gem") else SHAPE
    # one collider shape for both veils (each veil's collider knows only its own chains: map by bone name)
    cv = np.concatenate([c.v for c in colliders])
    off = np.cumsum([0] + [len(c.v) for c in colliders[:-1]])
    ct = np.concatenate([c.t + o for c, o in zip(colliders, off)])
    cbones = list(proxy.bones) + chain_bones
    cw = np.zeros((len(cv), len(cbones)))
    for c, o in zip(colliders, off):
        for j, b in enumerate(c.bones):
            cw[o:o + len(c.v), cbones.index(b)] += c.w[:, j]
    collider = Part("collider", "EV_PhysCloth", cv, ct, np.stack([cv[:, 0], cv[:, 2]], 1), "patch:edge", "collision",
                    cw, np.concatenate([c.D for c in colliders]))
    collider.bones = cbones
    collider.sliders = list(proxy.slider_names)
    pbody.sliders = list(proxy.slider_names)
    parts += [collider, pbody]
    # node hierarchy (root to both upper arms, then the chains) and the SMP system
    names = TORSO_CHAIN + arm_chain("L") + arm_chain("R")
    missing = [n for n in names if n not in skel]
    if missing:
        raise KeyError("skeleton nodes missing: %s" % missing)
    glob = {n: skel[n] for n in names}
    hierarchy = [(n, TORSO_CHAIN[i - 1] if i else None) for i, n in enumerate(TORSO_CHAIN)]
    for s in ("L", "R"):
        ch = arm_chain(s)
        hierarchy += [(n, ch[i - 1] if i else TORSO_CHAIN[-1]) for i, n in enumerate(ch)]
    for ch in chains:
        for n, m in zip(ch.names, ch.mats):
            glob[n] = m
        hierarchy += ch.hierarchy()
    kinematic = list(dict.fromkeys(pbody.bones))
    xml = mk_smp.system_xml(kinematic, chains, "EV_PhysBody", "EV_PhysCloth", mass=0.14, inertia=2.0,
                            lin_damp=0.55, ang_damp=0.7, gravity=0.8, swing=SWING_OUT, swing_in=SWING_IN,
                            side=SIDE_SWING, twist=0.15, stiffness=4.0, spring_damp=0.45, margin=0.32,
                            body_margin=0.3, penetration=0.5, wind=1.0, tag="evv")
    log("veils: %d parts, physics %d kinematic + %d chain bones" % (len(parts), len(kinematic), len(chain_bones)))
    return {"parts": parts, "fx": {"globals": glob, "hierarchy": hierarchy, "xml": xml, "chains": chains}}
