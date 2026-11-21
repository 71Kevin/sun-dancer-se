"""SMP (FSMP / HDT-SMP) physics for cloth panels: bone chains hung from a skeleton bone, panel weights blending from
the body to the chains where the cloth leaves the body, an invisible coarse collider that pushes the chains, a
decimated body to push against, and the system XML.

Conventions (the same as the Hollow Lantern tail, which works in game):
- every chain starts with a kinematic anchor ("... 00", mass 0) parented to a skeleton bone (the pelvis), followed
  by dynamic bones; a bone's local X runs down the chain, Y is the cloth's outward normal, Z the cloth's width;
- each dynamic bone is held to the previous one by a generic constraint with frameInB at the previous bone (the
  pivot is where the segment hangs from), with symmetric angular limits and a spring back to the rest pose;
- shapes in the XML: the collider (per-vertex, only the dynamic part of the cloth) and the body (per-triangle),
  both private to the system, so the cloth only collides with this outfit's own body proxy."""
import numpy as np

import mk_geom

SKELETON_CHAIN = ["NPC", "NPC Root [Root]", "NPC COM [COM ]", "CME Body [Body]", "CME LBody [LBody]",
                  "NPC Pelvis [Pelv]"]


def frame(origin, x_dir, y_hint):
    x = mk_geom.normalize(np.asarray(x_dir, dtype=np.float64))
    y = np.asarray(y_hint, dtype=np.float64)
    y = mk_geom.normalize(y - x * np.dot(x, y))
    z = np.cross(x, y)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, y, z, origin
    return m


class Chain:
    """A chain of bones along points P[0..n] (P[0] the kinematic anchor) with the cloth's outward normals N."""

    def __init__(self, names, P, N, parent="NPC Pelvis [Pelv]"):
        self.names = list(names)
        self.P = np.asarray(P, dtype=np.float64)
        self.parent = parent
        self.mats = []
        for k in range(len(P)):
            a, b = (k, k + 1) if k + 1 < len(P) else (k - 1, k)
            self.mats.append(frame(self.P[k], self.P[b] - self.P[a], N[k]))

    def hierarchy(self):
        return [(n, self.names[k - 1] if k else self.parent) for k, n in enumerate(self.names)]


def grid_rig(grid, contact, u, chain_u, nsegs, prefix, side_normal, parent="NPC Pelvis [Pelv]", blend_segs=0.6):
    """Chains for a cloth grid (rows x cols x 3, row 0 at the top) hanging from the body. contact (rows x cols): rows
    lying on the body; a chain's anchor sits where its column first leaves the body. Returns the chains and a
    function weights(rows_f, cols_f) -> (k, W) giving, for fractional grid coordinates, the blend factor k from body
    to chain (0 above the release line) and the weights over the chain bones (all chains' bones, in order)."""
    rows, cols = grid.shape[:2]
    chains, release = [], []
    for j, cu in enumerate(chain_u):
        c = int(round(cu * (cols - 1)))
        free = np.nonzero(~contact[:, c])[0]
        rr = int(free[0]) if len(free) else rows - 1
        rr = min(max(rr, 1), rows - 2)
        stations = rr + (rows - 1 - rr) * np.arange(nsegs + 1) / nsegs
        P, N = [], []
        for s in stations:
            r0 = int(np.floor(s))
            r1 = min(r0 + 1, rows - 1)
            f = s - r0
            P.append(grid[r0, c] * (1 - f) + grid[r1, c] * f)
            # outward normal of the cloth at that point (across and down the grid)
            down = grid[r1, c] - grid[max(r0 - 1, 0), c]
            across = grid[r0, min(c + 1, cols - 1)] - grid[r0, max(c - 1, 0)]
            n = mk_geom.normalize(np.cross(across, down))
            if np.dot(n, side_normal) < 0:
                n = -n
            N.append(n)
        names = ["%s%d %02d" % (prefix, j + 1, k) for k in range(nsegs + 1)]
        chains.append(Chain(names, np.array(P), np.array(N), parent))
        release.append((rr, (rows - 1 - rr) / nsegs))
    nb = len(chains) * (nsegs + 1)

    def weights(rf, cf):
        rf = np.asarray(rf, dtype=np.float64)
        uf = np.asarray(cf, dtype=np.float64) / (cols - 1)
        W = np.zeros((len(rf), nb))
        K = np.zeros(len(rf))
        cu = np.asarray(chain_u)
        j1 = np.clip(np.searchsorted(cu, uf), 1, len(cu) - 1) if len(cu) > 1 else np.zeros(len(uf), dtype=int)
        j0 = j1 - 1 if len(cu) > 1 else j1
        if len(cu) > 1:
            lat = np.clip((uf - cu[j0]) / np.maximum(cu[j1] - cu[j0], 1e-9), 0, 1)
        else:
            lat = np.zeros(len(uf))
        for js, wl in ((j0, 1 - lat), (j1, lat)):
            for i in range(len(rf)):
                j = int(js[i])
                rr, seg = release[j]
                p = np.clip((rf[i] - rr) / seg, 0, nsegs)
                k0 = int(min(np.floor(p), nsegs - 1))
                f = p - k0
                base = j * (nsegs + 1)
                W[i, base + k0] += wl[i] * (1 - f)
                W[i, base + k0 + 1] += wl[i] * f
                K[i] += wl[i] * np.clip((rf[i] - rr) / (blend_segs * seg), 0, 1)
        K = K * K * (3 - 2 * K)
        return K, W / np.maximum(W.sum(1, keepdims=True), 1e-9)
    return chains, weights


def coarse_collider(grid, weights, rows_at, cols_at):
    """Invisible low-resolution copy of the hanging cloth for the physics collision (a per-vertex shape: every vertex
    is a sphere pushing the bones it is weighted to). rows_at / cols_at: fractional grid coordinates."""
    rows, cols = grid.shape[:2]
    R, C = np.meshgrid(np.asarray(rows_at, dtype=np.float64), np.asarray(cols_at, dtype=np.float64), indexing="ij")
    rf, cf = R.ravel(), C.ravel()
    r0 = np.floor(rf).astype(int).clip(0, rows - 1)
    r1 = (r0 + 1).clip(0, rows - 1)
    c0 = np.floor(cf).astype(int).clip(0, cols - 1)
    c1 = (c0 + 1).clip(0, cols - 1)
    fr, fc = (rf - r0)[:, None], (cf - c0)[:, None]
    v = (grid[r0, c0] * (1 - fr) * (1 - fc) + grid[r1, c0] * fr * (1 - fc) + grid[r0, c1] * (1 - fr) * fc +
         grid[r1, c1] * fr * fc)
    t = mk_geom.grid_tris(len(rows_at), len(cols_at))
    _, W = weights(rf, cf)
    return v, t, W, (rf, cf)


def _vec(tag, x, y, z):
    return '<%s x="%g" y="%g" z="%g"/>' % (tag, x, y, z)


def system_xml(kinematic, chains, body_shape, cloth_shape, mass=0.4, inertia=6.0, lin_damp=0.45, ang_damp=0.55,
               gravity=1.0, swing=0.55, side=0.22, twist=0.08, stiffness=12.0, spring_damp=0.6, margin=0.45,
               body_margin=0.35, penetration=0.3, wind=0.5, tag="ev", swing_in=None):
    """The SMP system: kinematic bones (the body bones the collision body uses, and the chain anchors), dynamic chain
    bones getting lighter and freer toward the hem, chain constraints, the private collision shapes.

    swing_in: a smaller limit for the swing toward the body than away from it (None: symmetric). FSMP (hdtSMP64)
    builds each constraint with bodyA = the chain bone and bodyB = the one above it, and Bullet measures the angles as
    Euler XYZ of frameA^-1 * frameB: a positive Z angle is the chain bone turned toward its parent's -Y, i.e. toward
    the body (Y is the cloth's outward normal). So the upper Z limit holds the cloth off the body and the lower one
    lets it swing away."""
    z_lo, z_hi = -swing, (swing if swing_in is None else swing_in)
    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<system xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
           'xsi:noNamespaceSchemaLocation="description.xsd">',
           '\t<!-- Tinesh\'s Sun Dancer: cloth panels on bone chains (generated by tools/lib/mk_smp.py) -->']
    out.append('\t<bone-default>')
    out += ['\t\t<mass>0</mass>', '\t\t<inertia x="0" y="0" z="0"/>', '\t\t<margin-multiplier>1</margin-multiplier>',
            '\t</bone-default>']
    for b in kinematic:
        out.append('\t<bone name="%s"/>' % b)
    for ch in chains:
        out.append('\t<bone name="%s"/>' % ch.names[0])
    n = max(len(ch.names) for ch in chains) - 1
    for k in range(1, n + 1):
        f = (k - 1) / max(n - 1, 1)
        out += ['\t<bone-default>',
                '\t\t<mass>%.3f</mass>' % (mass * (1.0 - 0.35 * f)),
                '\t\t' + _vec("inertia", *(inertia * (1.0 - 0.4 * f),) * 3),
                '\t\t<centerOfMassTransform>', '\t\t\t<basis x="0" y="0" z="0" w="1"/>',
                '\t\t\t<origin x="0" y="0" z="0"/>', '\t\t</centerOfMassTransform>',
                '\t\t<linearDamping>%.3f</linearDamping>' % (lin_damp * (1.0 - 0.25 * f)),
                '\t\t<angularDamping>%.3f</angularDamping>' % (ang_damp * (1.0 - 0.25 * f)),
                '\t\t<friction>0.25</friction>', '\t\t<rollingFriction>0.05</rollingFriction>',
                '\t\t<restitution>0.05</restitution>', '\t\t<margin-multiplier>1</margin-multiplier>',
                '\t\t<gravity-factor>%.3f</gravity-factor>' % gravity,
                '\t\t<wind-factor>%.3f</wind-factor>' % (wind * (0.5 + 0.5 * f)),
                '\t</bone-default>']
        for ch in chains:
            if k < len(ch.names):
                out.append('\t<bone name="%s"/>' % ch.names[k])
    out += ['\t<generic-constraint-default>', '\t\t<frameInB>', '\t\t\t<basis x="0" y="0" z="0" w="1"/>',
            '\t\t\t<origin x="0" y="0" z="0"/>', '\t\t</frameInB>',
            '\t\t<useLinearReferenceFrameA>false</useLinearReferenceFrameA>',
            '\t\t' + _vec("linearLowerLimit", 0, 0, 0), '\t\t' + _vec("linearUpperLimit", 0, 0, 0),
            '\t\t' + _vec("angularLowerLimit", -twist, -side, z_lo),
            '\t\t' + _vec("angularUpperLimit", twist, side, z_hi),
            '\t\t' + _vec("linearStiffness", 0, 0, 0), '\t\t' + _vec("angularStiffness", stiffness, stiffness, stiffness),
            '\t\t' + _vec("linearDamping", 0, 0, 0), '\t\t' + _vec("angularDamping", spring_damp, spring_damp, spring_damp),
            '\t\t' + _vec("linearEquilibrium", 0, 0, 0), '\t\t' + _vec("angularEquilibrium", 0, 0, 0),
            '\t\t' + _vec("linearBounce", 0, 0, 0), '\t\t' + _vec("angularBounce", 0, 0, 0),
            '\t</generic-constraint-default>']
    for ch in chains:
        out.append('\t<constraint-group>')
        for k in range(1, len(ch.names)):
            out.append('\t\t<generic-constraint bodyA="%s" bodyB="%s"/>' % (ch.names[k], ch.names[k - 1]))
        out.append('\t</constraint-group>')
    out += ['\t<per-triangle-shape name="%s">' % body_shape, '\t\t<margin>%g</margin>' % body_margin,
            '\t\t<penetration>%g</penetration>' % penetration, '\t\t<shared>private</shared>',
            '\t\t<tag>%s_body</tag>' % tag, '\t\t<no-collide-with-tag>%s_body</no-collide-with-tag>' % tag,
            '\t</per-triangle-shape>',
            '\t<per-vertex-shape name="%s">' % cloth_shape, '\t\t<margin>%g</margin>' % margin,
            '\t\t<shared>private</shared>', '\t\t<tag>%s_cloth</tag>' % tag,
            '\t\t<can-collide-with-tag>%s_body</can-collide-with-tag>' % tag,
            '\t\t<no-collide-with-tag>%s_cloth</no-collide-with-tag>' % tag]
    for ch in chains:
        out.append('\t\t<weight-threshold bone="%s">1</weight-threshold>' % ch.names[0])
        for k in range(1, len(ch.names)):
            out.append('\t\t<weight-threshold bone="%s">0.2</weight-threshold>' % ch.names[k])
    out += ['\t</per-vertex-shape>', '</system>']
    return "\n".join(out) + "\n"
