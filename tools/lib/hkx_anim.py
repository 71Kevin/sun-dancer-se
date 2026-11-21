"""hkx_anim.py - reader of Havok 2010 packfiles (64-bit, Skyrim SE): the skeleton (hkaSkeleton) and
hkaSplineCompressedAnimation animations, with the pose of any frame and forward kinematics (each bone's world position).

  python hkx_anim.py info <file.hkx>                         objects and classes
  python hkx_anim.py pose <anim.hkx> <skeleton.hkx> [frame]  world positions of the main bones

Format as documented by HavokLib and hkxpack: blocks of B-splines with per-track masks; THREECOMP40/48 quaternions etc.
Used by the pose tests (tools/clip_test.py and tools/dev)."""
import math
import struct
import sys


class Packfile:
    def __init__(self, path):
        self.b = b = open(path, 'rb').read()
        if b[:8] != bytes.fromhex('57e0e05710c0c010'):
            raise ValueError('not a Havok packfile')
        if b[16] != 8:
            raise ValueError('only 64-bit packfiles are supported')
        nsec = struct.unpack_from('<i', b, 20)[0]
        self.sections = []
        for s in range(nsec):
            o = 0x40 + s * 0x30
            name = b[o:o + 19].split(b'\x00')[0].decode()
            vals = struct.unpack_from('<7I', b, o + 20)
            self.sections.append((name, vals))
        cn = next(v for n, v in self.sections if n == '__classnames__')
        self.classnames = {}
        start, end = cn[0], cn[0] + cn[1]
        i = start
        while i + 5 < end:
            if b[i + 4] != 0x09:
                break
            j = b.index(b'\x00', i + 5)
            self.classnames[i + 5 - start] = b[i + 5:j].decode()
            i = j + 1
        dname, d = next((n, v) for n, v in self.sections if n == '__data__')
        self.dsi = [n for n, _ in self.sections].index('__data__')
        self.base = d[0]
        lf, gf, vf, ex = d[1], d[2], d[3], d[4]
        self.local = {}
        for k in range(lf, gf, 8):
            src, dst = struct.unpack_from('<II', b, self.base + k)
            if src == 0xFFFFFFFF:
                continue
            self.local[src] = dst
        self.glob = {}
        for k in range(gf, vf, 12):
            src, sec, dst = struct.unpack_from('<III', b, self.base + k)
            if src == 0xFFFFFFFF:
                continue
            self.glob[src] = dst
        self.objects = []
        for k in range(vf, ex, 12):
            src, sec, cno = struct.unpack_from('<III', b, self.base + k)
            if src == 0xFFFFFFFF:
                continue
            self.objects.append((src, self.classnames.get(cno, '?')))

    def u(self, fmt, off):
        return struct.unpack_from('<' + fmt, self.b, self.base + off)

    def ptr(self, off):
        if off in self.local:
            return self.local[off]
        return self.glob.get(off)

    def array(self, off):
        """hkArray: (ptr, size) -> (data offset or None, size)"""
        size = self.u('i', off + 8)[0]
        return self.ptr(off), size

    def cstr(self, off):
        p = self.ptr(off)
        if p is None:
            return ''
        s = self.base + p
        return self.b[s:self.b.index(b'\x00', s)].decode('latin-1')

    def find(self, cls):
        return [o for o, c in self.objects if c == cls]


# ---------------- skeleton ----------------
class Skeleton:
    def __init__(self, path):
        pf = Packfile(path)
        o = pf.find('hkaSkeleton')[0]
        self.name = pf.cstr(o + 16)
        pp, n = pf.array(o + 24)
        self.parents = list(pf.u(f'{n}h', pp))
        bp, nb = pf.array(o + 40)
        self.names = [pf.cstr(bp + i * 16) for i in range(nb)]
        rp, nr = pf.array(o + 56)
        self.ref = []
        for i in range(nr):
            v = pf.u('12f', rp + i * 48)
            self.ref.append((v[0:3], v[4:8], v[8:11]))   # t, q(x,y,z,w), s
        self.index = {nm: i for i, nm in enumerate(self.names)}


# ---------------- quaternions / vectors ----------------
def qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


def qrot(q, v):
    x, y, z, w = q
    vx, vy, vz = v
    # v' = v + 2w(q x v) + 2 q x (q x v)
    cx = y * vz - z * vy
    cy = z * vx - x * vz
    cz = x * vy - y * vx
    c2x = y * cz - z * cy
    c2y = z * cx - x * cz
    c2z = x * cy - y * cx
    return (vx + 2 * (w * cx + c2x), vy + 2 * (w * cy + c2y), vz + 2 * (w * cz + c2z))


def qnorm(q):
    n = math.sqrt(sum(c * c for c in q)) or 1.0
    return tuple(c / n for c in q)


# ---------------- spline compressed ----------------
ROT_SIZE = {0: 4, 1: 5, 2: 6, 3: 3, 4: 2, 5: 16}
ROT_ALIGN = {0: 4, 1: 1, 2: 2, 3: 1, 4: 2, 5: 4}


def _reorder(vals, w, shift):
    x, y, z = vals
    if shift == 0:
        return (w, x, y, z)
    if shift == 1:
        return (x, w, y, z)
    if shift == 2:
        return (x, y, w, z)
    return (x, y, z, w)


def read_quat(b, i, qt):
    if qt == 1:   # THREECOMP40
        v = int.from_bytes(b[i:i + 5], 'little')
        m = (1 << 12) - 1
        fr = 0.000345436
        vals = [(((v >> (12 * k)) & m) - 2047) * fr for k in range(3)]
        shift = (v >> 36) & 3
        inv = (v >> 38) & 1
        w = math.sqrt(max(0.0, 1 - sum(c * c for c in vals)))
        if inv:
            w = -w
        return qnorm(_reorder(vals, w, shift))
    if qt == 2:   # THREECOMP48
        x, y, z = struct.unpack_from('<3H', b, i)
        m = (1 << 15) - 1
        fr = 0.000021585
        shift = ((y >> 14) & 2) | ((x >> 15) & 1)
        inv = (z >> 15) & 1
        vals = [((c & m) - 16383) * fr for c in (x, y, z)]
        w = math.sqrt(max(0.0, 1 - sum(c * c for c in vals)))
        if inv:
            w = -w
        return qnorm(_reorder(vals, w, shift))
    if qt == 5:   # UNCOMPRESSED
        return qnorm(struct.unpack_from('<4f', b, i))
    if qt == 0:   # POLAR32
        v = struct.unpack_from('<I', b, i)[0]
        rmask = (1 << 10) - 1
        rfrac = 1.0 / rmask
        phi_frac = math.pi / 4 / 511
        R = float((v >> 18) & rmask) * rfrac
        R = 1.0 - R * R
        phi_theta = float(v & 0x3FFFF)
        phi = math.floor(math.sqrt(phi_theta))
        theta = 0.0
        if phi > 0.0:
            theta = (math.pi / 4) * (phi_theta - phi * phi) / phi
            phi = phi_frac * phi
        magnitude = math.sqrt(max(0.0, 1.0 - R * R))
        x = math.sin(phi) * math.cos(theta) * magnitude
        y = math.sin(phi) * math.sin(theta) * magnitude
        z = math.cos(phi) * magnitude
        w = R
        if v & 0x10000000:
            x = -x
        if v & 0x20000000:
            y = -y
        if v & 0x40000000:
            z = -z
        if v & 0x80000000:
            w = -w
        return qnorm((x, y, z, w))
    raise NotImplementedError(f'rotation quantization {qt}')


def align(i, a):
    return (i + a - 1) // a * a


def find_span(degree, t, ncp, knots):
    if t >= knots[ncp]:
        return ncp - 1
    low, high = degree, ncp
    mid = (low + high) // 2
    guard = 0
    while t < knots[mid] or t >= knots[mid + 1]:
        if t < knots[mid]:
            high = mid
        else:
            low = mid
        mid = (low + high) // 2
        guard += 1
        if guard > 64:
            break
    return mid


def eval_spline(degree, knots, cps, t):
    ncp = len(cps)
    span = find_span(degree, t, ncp, knots)
    N = [1.0, 0.0, 0.0, 0.0, 0.0]
    for i in range(1, degree + 1):
        for j in range(i - 1, -1, -1):
            den = knots[span + i - j] - knots[span - j]
            A = (t - knots[span - j]) / den if den else 0.0
            tmp = N[j] * A
            N[j + 1] += tmp
            N[j] -= tmp
    dim = len(cps[0])
    out = [0.0] * dim
    for i in range(degree + 1):
        cp = cps[span - i]
        for d in range(dim):
            out[d] += cp[d] * N[i]
    return out


class Track:
    """one component (position, rotation or scale) of a track in a block"""
    def __init__(self):
        self.static = None
        self.spline = None   # (degree, knots, cps)

    def value(self, t):
        if self.spline:
            d, k, c = self.spline
            return eval_spline(d, k, c, t)
        return self.static


def read_vector(b, i, flags, qt, default):
    tr = Track()
    if flags & 0x70:
        nitems, degree = struct.unpack_from('<HB', b, i)
        i += 3
        knots = list(b[i:i + nitems + degree + 2])
        i += nitems + degree + 2
        i = align(i, 4)
        rng = []
        for ax in range(3):
            if flags & (0x10 << ax):
                mn, mx = struct.unpack_from('<2f', b, i)
                i += 8
                rng.append(('s', mn, mx))
            elif flags & (0x01 << ax):
                v = struct.unpack_from('<f', b, i)[0]
                i += 4
                rng.append(('c', v))
            else:
                rng.append(('c', default))
        cps = []
        for _ in range(nitems + 1):
            cp = []
            for r in rng:
                if r[0] == 's':
                    if qt == 0:
                        q = b[i] / 255.0
                        i += 1
                    else:
                        q = struct.unpack_from('<H', b, i)[0] / 65535.0
                        i += 2
                    cp.append(r[1] + (r[2] - r[1]) * q)
                else:
                    cp.append(r[1])
            cps.append(cp)
        i = align(i, 4)
        tr.spline = (degree, knots, cps)
    else:
        v = []
        for ax in range(3):
            if flags & (0x01 << ax):
                v.append(struct.unpack_from('<f', b, i)[0])
                i += 4
            else:
                v.append(default)
        i = align(i, 4)
        tr.static = v
    return tr, i


def read_rotation(b, i, flags, qt):
    tr = Track()
    if flags & 0xF0:
        nitems, degree = struct.unpack_from('<HB', b, i)
        i += 3
        knots = list(b[i:i + nitems + degree + 2])
        i += nitems + degree + 2
        i = align(i, ROT_ALIGN[qt])
        cps = []
        for _ in range(nitems + 1):
            cps.append(list(read_quat(b, i, qt)))
            i += ROT_SIZE[qt]
        # keep the sign continuous between control points
        for k in range(1, len(cps)):
            if sum(a * c for a, c in zip(cps[k - 1], cps[k])) < 0:
                cps[k] = [-c for c in cps[k]]
        tr.spline = (degree, knots, cps)
    elif flags & 0x0F:
        i = align(i, ROT_ALIGN[qt])
        tr.static = list(read_quat(b, i, qt))
        i += ROT_SIZE[qt]
    else:
        tr.static = [0.0, 0.0, 0.0, 1.0]
    i = align(i, 4)
    return tr, i


class Animation:
    def __init__(self, path):
        pf = Packfile(path)
        objs = pf.find('hkaSplineCompressedAnimation')
        if not objs:
            raise ValueError('no hkaSplineCompressedAnimation: ' + ', '.join(sorted({c for _, c in pf.objects})))
        o = objs[0]
        self.duration = pf.u('f', o + 20)[0]
        self.ntracks = pf.u('i', o + 24)[0]
        self.nframes, self.nblocks, self.maxfpb, self.mqsize = pf.u('4i', o + 56)
        self.block_dur, self.block_inv, self.frame_dur = pf.u('3f', o + 72)
        bo, nbo = pf.array(o + 88)
        self.block_offsets = list(pf.u(f'{nbo}I', bo))
        dp, nd = pf.array(o + 152)
        self.data = pf.b[pf.base + dp: pf.base + dp + nd]
        bind = pf.find('hkaAnimationBinding')
        self.track_to_bone = None
        if bind:
            tp, nt = pf.array(bind[0] + 32)
            if tp is not None and nt:
                self.track_to_bone = list(pf.u(f'{nt}h', tp))
        self._blocks = {}

    def block(self, bi):
        if bi in self._blocks:
            return self._blocks[bi]
        b = self.data
        start = self.block_offsets[bi]
        masks = [b[start + 4 * t: start + 4 * t + 4] for t in range(self.ntracks)]
        i = start + self.mqsize
        tracks = []
        for t in range(self.ntracks):
            qtypes, ptypes, rtypes, stypes = masks[t]
            pq, rq, sq = qtypes & 3, (qtypes >> 2) & 0xF, (qtypes >> 6) & 3
            pos, i = read_vector(b, i, ptypes, pq, 0.0)
            rot, i = read_rotation(b, i, rtypes, rq)
            scl, i = read_vector(b, i, stypes, sq, 1.0)
            tracks.append((pos, rot, scl))
        self._blocks[bi] = tracks
        return tracks

    def pose(self, frame):
        frame = max(0, min(frame, self.nframes - 1))
        bi = min(int(frame // (self.maxfpb - 1)) if self.maxfpb > 1 else 0, self.nblocks - 1)
        t = frame - bi * (self.maxfpb - 1)
        out = []
        for pos, rot, scl in self.block(bi):
            q = qnorm(rot.value(t))
            out.append((tuple(pos.value(t)), q, tuple(scl.value(t))))
        return out


def world_pose(skel, anim_pose, track_to_bone=None):
    n = len(skel.names)
    local = list(skel.ref)
    for ti, tr in enumerate(anim_pose):
        bone = track_to_bone[ti] if track_to_bone else ti
        if 0 <= bone < n:
            local[bone] = tr
    world = [None] * n
    for i in range(n):
        t, q, s = local[i]
        p = skel.parents[i]
        if p < 0:
            world[i] = (tuple(t), tuple(q))
        else:
            pt, pq = world[p]
            r = qrot(pq, t)
            world[i] = ((pt[0] + r[0], pt[1] + r[1], pt[2] + r[2]), qnorm(qmul(pq, q)))
    return world


def main():
    cmd = sys.argv[1]
    if cmd == 'info':
        pf = Packfile(sys.argv[2])
        for o, c in pf.objects:
            print(hex(o), c)
    elif cmd == 'pose':
        an = Animation(sys.argv[2])
        sk = Skeleton(sys.argv[3])
        fr = int(sys.argv[4]) if len(sys.argv) > 4 else 0
        print(f'frames {an.nframes} blocks {an.nblocks} tracks {an.ntracks} dur {an.duration:.2f}s | skeleton {len(sk.names)} bones')
        w = world_pose(sk, an.pose(fr), an.track_to_bone)
        for nm in ('NPC Root [Root]', 'NPC Pelvis [Pelv]', 'NPC Spine2 [Spn2]', 'NPC Head [Head]', 'NPC L Hand [LHnd]', 'NPC R Hand [RHnd]',
                   'NPC L Forearm [LLar]', 'NPC R Forearm [RLar]', 'NPC L Foot [Lft ]', 'NPC R Foot [Rft ]'):
            if nm in sk.index:
                p = w[sk.index[nm]][0]
                print(f'  {nm:24s} {p[0]:8.2f} {p[1]:8.2f} {p[2]:8.2f}')


if __name__ == '__main__':
    main()
