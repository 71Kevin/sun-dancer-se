import numpy as np

import mk_blend
import mk_geom


class Body:
    def __init__(self, ref, shape_names=("BaseShape",), weld_tol=1e-4):
        shapes = [ref["shapes"][n] for n in shape_names]
        self.shape_names = list(shape_names)
        bones = []
        for sh in shapes:
            for b in sh["bones"]:
                if b not in bones:
                    bones.append(b)
        self.bones = bones
        s2b = {}
        for sh in shapes:
            for j, b in enumerate(sh["bones"]):
                s2b.setdefault(b, sh["s2b"][j])
        self.s2b = np.array([s2b[b] for b in bones])
        vs, ts, ws = [], [], []
        self.ranges = {}
        off = 0
        for name, sh in zip(shape_names, shapes):
            n = len(sh["v"])
            self.ranges[name] = (off, off + n)
            vs.append(sh["v"])
            ts.append(sh["t"] + off)
            w = np.zeros((n, len(bones)), dtype=np.float64)
            for j, b in enumerate(sh["bones"]):
                w[:, bones.index(b)] = sh["w"][:, j]
            ws.append(w)
            off += n
        v_all = np.concatenate(vs)
        t_all = np.concatenate(ts)
        w_all = np.concatenate(ws)
        names = []
        self.uv_attrs = []
        for attrs, data in ref["sliders"]:
            if not any(tgt in shape_names for _, tgt, _ in data):
                continue
            if attrs.get("uv") == "true":   # texture-coordinate sliders: only the skin shape carries them
                if attrs not in self.uv_attrs:
                    self.uv_attrs.append(attrs)
                continue
            if attrs["name"] not in names:
                names.append(attrs["name"])
        self.slider_names = names
        self.slider_attrs = []
        D_all = np.zeros((len(v_all), len(names), 3), dtype=np.float32)
        for attrs, data in ref["sliders"]:
            if attrs["name"] not in names:
                continue
            if attrs not in self.slider_attrs:
                self.slider_attrs.append(attrs)
            k = names.index(attrs["name"])
            for dname, tgt, _ in data:
                if tgt not in shape_names:
                    continue
                idx, d = ref["diffs"][(attrs["name"], tgt)]
                if len(idx):
                    D_all[idx + self.ranges[tgt][0], k] = d
        q = np.round(v_all / weld_tol).astype(np.int64)
        _, first, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
        inv = inv.ravel()
        cnt = np.bincount(inv).astype(np.float64)
        self.inv = inv
        self.v = v_all[first]
        t = inv[t_all]
        good = (t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 2] != t[:, 0])
        self.t = t[good]
        w = np.zeros((len(first), len(bones)))
        np.add.at(w, inv, w_all)
        self.w = w / cnt[:, None]
        D = np.zeros((len(first), len(names), 3), dtype=np.float64)
        np.add.at(D, inv, D_all)
        self.D = (D / cnt[:, None, None]).astype(np.float32)
        self.orig_t = {name: sh["t"] + self.ranges[name][0] for name, sh in zip(shape_names, shapes)}
        self.n = mk_geom.vertex_normals(self.v, self.t)
        self.tree = mk_blend.bvh(self.v, self.t)

    def bone_globals(self):
        return {b: np.linalg.inv(self.s2b[j]) for j, b in enumerate(self.bones)}

    def mask_bones(self, keys):
        m = np.zeros(len(self.v))
        for j, b in enumerate(self.bones):
            if any(k in b for k in keys):
                m += self.w[:, j]
        return m

    def signed_distance(self, points):
        loc, nrm, face, dist = mk_blend.nearest(self.tree, points)
        s = np.sign(((points - loc) * nrm).sum(1))
        s[s == 0] = 1
        return s * dist, loc, nrm, face

    def project(self, points):
        loc, nrm, face, dist = mk_blend.nearest(self.tree, points)
        tri = self.t[face]
        bc = mk_blend.barycentric(loc, self.v[tri[:, 0]], self.v[tri[:, 1]], self.v[tri[:, 2]])
        return tri, bc, loc, nrm, dist

    def sample(self, tri, bc, array):
        a = np.asarray(array)
        return np.einsum("nk,nk...->n...", bc, a[tri])

    def smoothed(self, mask, iters=30, lam=0.5, mu=-0.53):
        free = np.asarray(mask, dtype=np.float64)
        pairs = mk_geom.sub_pairs(mk_geom.neighbor_pairs(self.t, len(self.v)), free > 0)
        rows = pairs[3]
        v = self.v.copy()
        D = self.D.reshape(len(self.v), -1).astype(np.float64)
        for _ in range(iters):
            for f in (lam, mu):
                k = (f * free[rows])[:, None]
                v[rows] += k * mk_geom.laplacian(v, pairs)[rows]
                D[rows] += k * mk_geom.laplacian(D, pairs)[rows]
        out = object.__new__(Body)
        out.__dict__.update(self.__dict__)
        out.v = v
        out.D = D.reshape(self.D.shape).astype(np.float32)
        out.n = mk_geom.vertex_normals(v, self.t)
        out.tree = mk_blend.bvh(v, self.t)
        return out

    def apply_preset(self, values):
        out = self.v.copy()
        for k, name in enumerate(self.slider_names):
            if name in values:
                out += self.D[:, k] * values[name]
        return out


def sym_curve(ctrl, samples=16):
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


def push_out(points, body, offset, iters=2):
    p = np.array(points, dtype=np.float64)
    off = np.broadcast_to(np.asarray(offset, dtype=np.float64), (len(p),))
    for _ in range(iters):
        sd, loc, nrm, _ = body.signed_distance(p)
        bad = sd < off
        if not bad.any():
            break
        p[bad] = loc[bad] + nrm[bad] * off[bad, None]
    return p


def drape(points, tris, body, offset, iters=40, lam=0.5, fixed=None, pull=0.0, check_every=2):
    pairs = mk_geom.neighbor_pairs(tris, len(points))
    p = push_out(points, body, offset)
    for k in range(iters):
        dp = lam * mk_geom.laplacian(p, pairs)
        if fixed is not None:
            dp[fixed] = 0
        p = p + dp
        if pull:
            sd, loc, nrm, _ = body.signed_distance(p)
            off = np.broadcast_to(np.asarray(offset, dtype=np.float64), (len(p),))
            target = loc + nrm * off[:, None]
            p = p + pull * (target - p) * (sd > off)[:, None]
        if k % check_every == check_every - 1 or k == iters - 1:
            p = push_out(p, body, offset, iters=1)
    return push_out(p, body, offset, iters=2)


def transfer(points, body, smooth_tris=None, smooth_iters=0, bone_filter=None, weight_scale=None):
    tri, bc, loc, nrm, dist = body.project(points)
    w = body.sample(tri, bc, body.w)
    if bone_filter is not None:
        w = w * bone_filter[None, :]
    if weight_scale is not None:
        w = w * weight_scale[None, :]
    D = body.sample(tri, bc, body.D.reshape(len(body.v), -1)).reshape(len(points), -1, 3)
    if smooth_tris is not None and smooth_iters:
        pairs = mk_geom.neighbor_pairs(smooth_tris, len(points))
        w = mk_geom.smooth(w, pairs, smooth_iters, 0.5)
        D = mk_geom.smooth(D.reshape(len(points), -1), pairs, smooth_iters, 0.5).reshape(D.shape)
    s = w.sum(1, keepdims=True)
    s[s == 0] = 1
    return w / s, D, dist


def harmonic_fill(tris, X, k, iters=600):
    X = np.asarray(X, dtype=np.float64)
    flat = X.reshape(len(X), -1).copy()
    sel = k > 0.999
    rows = np.nonzero(sel)[0]
    sub = mk_geom.sub_pairs(mk_geom.neighbor_pairs(tris, len(X)), sel)
    for _ in range(iters):
        flat[rows] += 0.9 * mk_geom.laplacian(flat, sub)[rows]
    kk = k.reshape((-1,) + (1,) * (X.ndim - 1))
    return X * (1 - kk) + flat.reshape(X.shape) * kk


def beneath_weights(points, tris, body, bone_mask=None, reach=3.0, smooth_iters=4):
    n = mk_geom.vertex_normals(points, tris)
    loc, nrm, face, dist = mk_blend.raycast(body.tree, points, -n, reach)
    miss = ~np.isfinite(dist)
    if miss.any():
        loc2, _, face2, _ = mk_blend.nearest(body.tree, points[miss])
        loc[miss] = loc2
        face[miss] = face2
    tri = body.t[face]
    bc = mk_blend.barycentric(loc, body.v[tri[:, 0]], body.v[tri[:, 1]], body.v[tri[:, 2]])
    w = np.einsum("nk,nkb->nb", bc, body.w[tri])
    if bone_mask is not None:
        w = w * bone_mask[None, :]
    if smooth_iters:
        w = mk_geom.smooth(w, mk_geom.neighbor_pairs(tris, len(points)), smooth_iters, 0.5)
    return w / np.maximum(w.sum(1, keepdims=True), 1e-9)


def garment_conform(points, D, body, slack_margin=None, normals=None, smooth_tris=None, smooth_iters=60,
                    along_normals=False, min_cos=0.3):
    """Per slider, the garment moves out at least as far as the body under it. along_normals: the body's motion is
    measured along the garment's own normals and the push goes along them (hanging cloth: only forward or back).
    normals (without along_normals): the push goes along the garment's normals, lengthened by 1/cos of their angle
    to the body's normal (cos clipped at min_cos)."""
    tri, bc, loc, nrm, dist = body.project(points)
    Db = body.sample(tri, bc, body.D.reshape(len(body.v), -1)).reshape(D.shape)
    if along_normals:
        nrm = normals
    gap = ((Db - D) * nrm[:, None, :]).sum(2)
    if slack_margin is not None:
        gap -= np.clip(dist - slack_margin, 0, None)[:, None]
    gap = np.clip(gap, 0, None)
    if smooth_tris is not None:
        pairs = mk_geom.neighbor_pairs(smooth_tris, len(points))
        need = gap
        for _ in range(smooth_iters):
            gap = np.maximum(need, gap + 0.5 * mk_geom.laplacian(gap, pairs))
    if normals is None or along_normals:
        return D + gap[:, :, None] * nrm[:, None, :]
    cos = np.clip((normals * nrm).sum(1), min_cos, 1.0)
    return D + (gap / cos[:, None])[:, :, None] * normals[:, None, :]


def standoff_smooth(points, tris, D, body, start=0.9, iters=30):
    _, _, _, _, dist = body.project(points)
    pairs = mk_geom.neighbor_pairs(tris, len(points))
    flat = D.reshape(len(points), -1)
    k = np.clip((dist - start) / 0.6, 0, 1)
    for _ in range(iters):
        flat = flat + 0.5 * k[:, None] * mk_geom.laplacian(flat, pairs)
    return flat.reshape(D.shape)


def welded(v, t, tol=1e-4):
    q = np.round(np.asarray(v) / tol).astype(np.int64)
    _, first, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    return np.asarray(v, dtype=np.float64)[first], inv[t], first, inv


def layer_body(v, t, D):
    out = object.__new__(Body)
    out.v = np.asarray(v, dtype=np.float64)
    out.t = np.asarray(t)
    out.D = np.asarray(D, dtype=np.float32)
    out.n = mk_geom.vertex_normals(out.v, out.t)
    out.tree = mk_blend.bvh(out.v, out.t)
    return out


def fill_concave(v, t, k, iters=220, lam=0.5):
    pairs = mk_geom.neighbor_pairs(t, len(v))
    p = np.array(v, dtype=np.float64)
    for i in range(iters):
        if i % 4 == 0:
            n = mk_geom.vertex_normals(p, t)
        dp = lam * mk_geom.laplacian(p, pairs)
        dn = (dp * n).sum(1)
        dp -= np.minimum(dn, 0)[:, None] * n
        p += dp * k[:, None]
    return p


def random_presets(count, n_sliders, seed=11, active=0.35, lo=-1.1, hi=1.4):
    rng = np.random.default_rng(seed)
    on = rng.random((count, n_sliders)) < active
    return rng.uniform(lo, hi, (count, n_sliders)) * on


def fold_safe_diffs(v, t, D, sliders, skip=(), damp=None, samples=160, rounds=8, iters=30, grow=3,
                    min_cos=0.0, min_area=0.2):
    wv, wt, first, inv = welded(v, t)
    pairs = mk_geom.neighbor_pairs(wt, len(wv))
    i, j = pairs[0], pairs[1]
    f0 = mk_geom.face_normals(wv, wt)
    a0 = np.linalg.norm(f0, axis=1)
    n0 = f0 / np.maximum(a0, 1e-12)[:, None]
    real = a0 > 0.25 * np.median(a0)
    out = np.array(D, dtype=np.float64)
    cols = [k for k, n in enumerate(sliders) if not any(s in n for s in skip) and np.abs(out[:, k]).max() > 1e-3]
    Dw = out[first][:, cols]
    vals = random_presets(samples, len(cols))
    scale = np.ones(len(wv))
    if damp is not None:
        np.minimum.at(scale, inv, damp)
    mask = np.zeros(len(wv))
    history = []
    for r in range(rounds + 1):
        hits = np.zeros(len(wv))
        bad_total = 0
        for s in range(samples):
            x = wv + np.einsum("c,ncd->nd", vals[s], Dw)
            f = mk_geom.face_normals(x, wt)
            a = np.linalg.norm(f, axis=1)
            bad = (((f * n0).sum(1) <= min_cos * a) | (a < min_area * a0)) & real
            bad_total += bad.sum()
            hits[wt[bad].ravel()] += 1
        history.append(bad_total / samples)
        if r == rounds or bad_total == 0:
            break
        m = (hits > 0).astype(np.float64)
        for _ in range(grow):
            g = m.copy()
            np.maximum.at(g, i, m[j])
            m = g
        mask = np.maximum(mask, m * scale)
        sub = mk_geom.sub_pairs(pairs, mask > 0)
        rows = sub[3]
        flat = Dw.reshape(len(wv), -1)
        for _ in range(iters):
            flat[rows] += 0.5 * mask[rows, None] * mk_geom.laplacian(flat, sub)[rows]
        Dw = flat.reshape(Dw.shape)
    out[:, cols] = Dw[inv]
    return out, history


def torso_axis(body, z_values):
    arm = body.mask_bones(("UpperArm", "Forearm", "Hand", "Clavicle", "Finger"))
    legs = body.mask_bones(("Thigh", "Calf", "Foot"))
    m = (arm < 0.05)
    cy = []
    for z in z_values:
        sel = m & (np.abs(body.v[:, 2] - z) < 0.8) & (np.abs(body.v[:, 0]) < 10)
        if sel.sum() < 4:
            cy.append(np.nan)
            continue
        p = body.v[sel]
        cy.append(0.5 * (p[:, 1].min() + p[:, 1].max()))
    cy = np.array(cy)
    good = ~np.isnan(cy)
    cy = np.interp(z_values, np.asarray(z_values)[good], cy[good])
    return cy


def ring_grid(body, z_values, n_theta, center_y, theta0=0.0):
    rows = len(z_values)
    th = theta0 + np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    origins = np.zeros((rows * n_theta, 3))
    dirs = np.zeros((rows * n_theta, 3))
    for r, z in enumerate(z_values):
        origins[r * n_theta:(r + 1) * n_theta] = [0.0, center_y[r], z]
        dirs[r * n_theta:(r + 1) * n_theta, 0] = np.sin(th)
        dirs[r * n_theta:(r + 1) * n_theta, 1] = np.cos(th)
    loc, nrm, face, dist = mk_blend.raycast(body.tree, origins, dirs, 60.0)
    miss = np.isnan(loc[:, 0])
    if miss.any():
        loc[miss] = origins[miss] + dirs[miss] * 10.0
    theta = np.tile(th, rows)
    zz = np.repeat(z_values, n_theta)
    t = mk_geom.grid_tris(rows, n_theta, wrap=True)[:, ::-1].copy()
    return loc, t, theta, zz


def conform_samples(body, D, tris, idx, garment_parts, samples, skip=(), rounds=4, margin=0.04, reach=1.5):
    vs, ts, ds = [], [], []
    off = 0
    for p in garment_parts:
        vs.append(p.v)
        ts.append(p.t + off)
        ds.append(np.asarray(p.D, dtype=np.float64))
        off += len(p.v)
    gv = np.concatenate(vs)
    gt = np.concatenate(ts)
    gD = np.concatenate(ds)
    names = body.slider_names
    S = np.array([[vals.get(n, 0.0) for n in names] for vals in samples])
    G = S * np.array([not any(k in n for k in skip) for n in names])
    out = np.array(D, dtype=np.float64)
    history = []
    active = np.arange(len(S))
    for r in range(rounds + 1):
        need = np.zeros(len(idx))
        dirs = np.zeros((len(idx), 3))
        pick = np.full(len(idx), -1)
        total = 0
        poked = []
        for s in active:
            bv = body.v + np.einsum("j,njd->nd", S[s], out)
            bn = mk_geom.vertex_normals(bv, tris)[idx]
            tree = mk_blend.bvh(gv + np.einsum("j,njd->nd", G[s], gD), gt)
            loc, nrm, face, dist = mk_blend.raycast(tree, bv[idx] + bn * 0.005, -bn, reach)
            poke = np.isfinite(dist) & ((nrm * bn).sum(1) > 0)
            total += poke.sum()
            if poke.any():
                poked.append(s)
            c = np.where(poke, dist + margin, 0.0)
            better = c > need
            need[better] = c[better]
            dirs[better] = bn[better]
            pick[better] = s
        history.append(total / len(S))
        if not poked:
            if len(active) == len(S):
                break
            active = np.arange(len(S))
            continue
        if r == rounds:
            break
        active = np.array(poked)
        hit = pick >= 0
        sv = S[pick[hit]]
        k = need[hit] / np.maximum((sv * sv).sum(1), 1e-9)
        out[idx[hit]] -= (k[:, None] * sv)[:, :, None] * dirs[hit][:, None, :]
    return out, history


def conform_diffs(body, D, idx, garment_parts, uv_sliders=(), reach=4.0):
    vs, ts, ds = [], [], []
    off = 0
    for p in garment_parts:
        vs.append(p.v)
        ts.append(p.t + off)
        ds.append(np.asarray(p.D, dtype=np.float64))
        off += len(p.v)
    gv = np.concatenate(vs)
    gt = np.concatenate(ts)
    gD = np.concatenate(ds)
    tree = mk_blend.bvh(gv, gt)
    n = body.n[idx]
    loc, nrm, face, dist = mk_blend.raycast(tree, body.v[idx] + n * 0.005, n, reach)
    hit = np.isfinite(dist) & ((nrm * n).sum(1) > 0)
    rows = idx[hit]
    tri = gt[face[hit]]
    bc = mk_blend.barycentric(loc[hit], gv[tri[:, 0]], gv[tri[:, 1]], gv[tri[:, 2]])
    g = np.einsum("nk,nksd->nsd", bc, gD[tri])
    nn = n[hit]
    out = np.array(D, dtype=np.float64)
    skip = np.array([s in uv_sliders for s in body.slider_names])
    rel = out[rows] - g
    r = np.einsum("nsd,nd->ns", rel, nn)
    r[:, skip] = 0
    out[rows] -= np.clip(r, 0, None)[:, :, None] * nn[:, None, :]
    return out, len(rows)


# ---------------------------------------------------------------- rigid rings, surface turn, seams
def axial_profile(body, a, b, s_values, radius=4.0, band=0.4, bone_filter=None, passes=4):
    """Bone weights of a ring that turns as a whole around the limb a->b: at each distance s along the axis, the
    body's weights averaged over the limb's cross-section (vertices within `band` of the ring plane and `radius` of
    the axis), smoothed along the axis. A band with these weights twists along its length like a sleeve instead of
    shearing around its circumference (the forearm twist bones turn the skin by a different angle on each side)."""
    a = np.asarray(a, dtype=np.float64)
    ax = mk_geom.normalize(np.asarray(b, dtype=np.float64) - a)
    d = body.v - a
    h = d @ ax
    r = np.linalg.norm(d - np.outer(h, ax), axis=1)
    W = body.w if bone_filter is None else body.w * bone_filter[None, :]
    out = np.zeros((len(s_values), W.shape[1]))
    for k, s in enumerate(s_values):
        m = (np.abs(h - s) < band) & (r < radius)
        if m.sum() < 4:
            m = np.argsort(np.abs(h - s) + 100.0 * (r > radius))[:24]
        out[k] = W[m].mean(0)
    for _ in range(passes):
        out[1:-1] = 0.25 * out[:-2] + 0.5 * out[1:-1] + 0.25 * out[2:]
    return out / np.maximum(out.sum(1, keepdims=True), 1e-9)


def axial_weights(points, a, b, s_values, table):
    """Per-point weights interpolated from an axial_profile table by the point's distance along a->b."""
    a = np.asarray(a, dtype=np.float64)
    ax = mk_geom.normalize(np.asarray(b, dtype=np.float64) - a)
    s = (np.asarray(points) - a) @ ax
    s = np.clip(s, s_values[0], s_values[-1])
    k = np.clip(np.searchsorted(s_values, s) - 1, 0, len(s_values) - 2)
    f = ((s - s_values[k]) / np.maximum(s_values[k + 1] - s_values[k], 1e-9))[:, None]
    w = table[k] * (1 - f) + table[k + 1] * f
    return w / np.maximum(w.sum(1, keepdims=True), 1e-9)


def normal_turn_diffs(points, D, body, min_mag=0.05, tris=None):
    """Adds to a garment's slider data the turn of the body surface under it: a garment `off` above the body keeps
    that distance along the body's new normal when a slider bends the surface (the big breast sliders turn the lower
    breast from facing forward to facing down; with a constant offset the cups sank into it). Linear per slider, like
    all slider data."""
    tri, bc, loc, nrm, dist = body.project(points)
    off = ((np.asarray(points) - loc) * nrm).sum(1)
    t = body.t if tris is None else tris
    n0 = mk_geom.normalize(body.sample(tri, bc, mk_geom.vertex_normals(body.v, t)))
    out = np.array(D, dtype=np.float64)
    region = np.unique(tri)
    for k in range(len(body.slider_names)):
        dk = body.D[:, k]
        if np.abs(dk[region]).max(initial=0) < min_mag:
            continue
        nk = mk_geom.normalize(body.sample(tri, bc, mk_geom.vertex_normals(body.v + dk, t)))
        out[:, k] += off[:, None] * (nk - n0)
    return out.astype(np.asarray(D).dtype)


def weld_attributes(v, arrays, tol=1e-4):
    """Coincident vertices (seams of opened rings, island cuts, layer rims) get the mean of their weights and slider
    data, so the seams cannot open when the mesh is skinned or morphed."""
    q = np.round(np.asarray(v) / tol).astype(np.int64)
    _, inv, cnt = np.unique(q, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    if (cnt > 1).sum() == 0:
        return arrays
    out = []
    for a in arrays:
        a = np.asarray(a)
        flat = a.reshape(len(a), -1).astype(np.float64)
        acc = np.zeros((len(cnt), flat.shape[1]))
        np.add.at(acc, inv, flat)
        mean = acc / cnt[:, None]
        res = np.where((cnt[inv] > 1)[:, None], mean[inv], flat)
        out.append(res.reshape(a.shape).astype(a.dtype))
    return out


def _top_mask(w, k):
    order = np.argsort(-w, axis=1)
    keep = np.zeros(w.shape, dtype=bool)
    np.put_along_axis(keep, order[:, :k], True, axis=1)
    return keep


def limit_influences(w, t, max_inf=4, tol=0.02, rings=3, iters=400, lam=0.5):
    """Bone weights with at most max_inf bones per vertex that still vary continuously over the mesh.

    A NIF keeps 4 bones per vertex. Cutting the others vertex by vertex (mk_nif.limit_weights) drops a bone at one
    vertex and keeps it at the next; when that bone turns, the two vertices part: a crease and a seam of shading along
    a staircase of triangle edges (the physics chains of the loincloth's back panel showed one in game, where the
    panel's skin weights, the chain anchors and two chains' bones met). Here a bone that has to go fades out: where it
    is cut, its weight is relaxed toward zero over the few rings of vertices around the cut, the weights are
    renormalised and the kept bones chosen again, until no vertex loses more than tol. Returns (weights, report)."""
    w = np.array(w, dtype=np.float64)
    w /= np.maximum(w.sum(1, keepdims=True), 1e-12)
    n = len(w)
    i, j, deg, _, _ = mk_geom.neighbor_pairs(t, n)
    deg = np.maximum(deg, 1.0)
    first = float(w[~_top_mask(w, max_inf) & (w > 0)].max(initial=0.0))
    steps = 0
    for steps in range(1, iters + 1):
        keep = _top_mask(w, max_inf)
        cut = ~keep & (w > 0)
        if not cut.any() or w[cut].max() < tol:
            break
        w[cut] = 0.0
        for b in np.nonzero(cut.any(0))[0]:
            band = cut[:, b].copy()
            for _ in range(rings):
                grow = band.copy()
                np.logical_or.at(grow, i, band[j])
                band = grow
            band &= w[:, b] > 0
            if not band.any():
                continue
            x = w[:, b].copy()
            for _ in range(2 * rings):
                avg = np.zeros(n)
                np.add.at(avg, i, x[j])
                x = np.where(band, x + lam * (avg / deg - x), x)
            w[:, b] = x
        w /= np.maximum(w.sum(1, keepdims=True), 1e-12)
    keep = _top_mask(w, max_inf)
    last = float(w[~keep & (w > 0)].max(initial=0.0))
    w = np.where(keep, w, 0.0)
    w /= np.maximum(w.sum(1, keepdims=True), 1e-12)
    return w, {"cut_before": first, "cut_after": last, "steps": steps}


def mirror_bones(bones):
    """Index of each bone's left/right twin ("NPC L Butt" <-> "NPC R Butt", "[LThg]" <-> "[RThg]"); itself if none."""
    import re
    names = list(bones)
    out = []
    for b in names:
        m = re.sub(r"\bL\b", "\x00", b)
        m = re.sub(r"\bR\b", "L", m).replace("\x00", "R")
        m = re.sub(r"\[L", "\x01", m)
        m = re.sub(r"\[R", "[L", m).replace("\x01", "[R")
        out.append(names.index(m) if m in names else names.index(b))
    return np.array(out)


def blend_midline(w, bones, x, width=1.5):
    """Weights that cross the body's midline smoothly: toward x = 0 each vertex blends into the mean of its weights
    and their left/right mirror. A garment bridging the buttock cleft took the skin's weights, which switch from the
    left buttock to the right one at the cleft; when the buttocks moved, the two halves of the cloth turned apart and
    the shading changed sharply down the middle (a light block on one side of the back panel in game)."""
    w = np.asarray(w, dtype=np.float64)
    k = np.clip(1.0 - np.abs(np.asarray(x)) / width, 0, 1)
    k = k * k * (3 - 2 * k)
    sym = 0.5 * (w + w[:, mirror_bones(bones)])
    out = w + k[:, None] * (sym - w)
    return out / np.maximum(out.sum(1, keepdims=True), 1e-12)


def edge_jumps(w, t):
    """Largest change of the bone weights across one edge (half the L1 distance of the two weight rows)."""
    e = mk_geom.unique_edges(t)
    return 0.5 * np.abs(np.asarray(w)[e[:, 0]] - np.asarray(w)[e[:, 1]]).sum(1)
