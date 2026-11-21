"""Sun Dancer Circlet (slot 42): a slender gold band around the head, close to the forehead and riding over
the hair at the sides and back, with a sun disc and a hanging amber drop on the forehead. Rigid on the head bone."""
import os
import pickle

import numpy as np

import mk_blend
import mk_fit
import mk_geom
from mk_parts import Part
from ev_shapes import teardrop
from piece_wrap import dome, ribbon, sun_disc

FRONT_Z = 125.6
BACK_Z = 128.6
OFF_FRONT = 0.32
OFF_BACK = 1.15
WIDTH = 0.5
THICK = 0.14


def head_layer(work):
    h = pickle.load(open(os.path.join(work, "ref", "head.pkl"), "rb"))["shape"]
    v, t = h["v"], h["t"]
    q = np.round(v / 1e-4).astype(np.int64)
    _, first, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    tt = inv[t]
    good = (tt[:, 0] != tt[:, 1]) & (tt[:, 1] != tt[:, 2]) & (tt[:, 2] != tt[:, 0])
    return mk_fit.layer_body(v[first], tt[good], np.zeros((len(first), 1, 3), dtype=np.float32)), h


def build(body, proxy, log=print, work=None):
    work = work or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                "cache")
    head, href = head_layer(work)
    c = np.array([0.0, 0.6, 0.5 * (FRONT_Z + BACK_Z)])
    ang = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    # ring in a plane tilted down toward the forehead
    pts, nrms = [], []
    for a in ang:
        front = 0.5 * (1 + np.cos(a))
        z = BACK_Z + (FRONT_Z - BACK_Z) * front
        d = np.array([np.sin(a), np.cos(a), 0.0])
        o = np.array([c[0], c[1], z])
        loc, nrm, _, dist = mk_blend.raycast(head.tree, o[None], d[None], 20.0)
        if not np.isfinite(dist[0]):
            continue
        off = OFF_BACK + (OFF_FRONT - OFF_BACK) * front ** 2
        n = nrm[0] if np.dot(nrm[0], d) > 0 else -nrm[0]
        n = mk_geom.normalize(n * np.array([1, 1, 0.35]))
        pts.append(loc[0] + n * off)
        nrms.append(n)
    pts = np.array(pts)
    nrms = np.array(nrms)
    for _ in range(8):
        pts = 0.5 * pts + 0.25 * (np.roll(pts, 1, 0) + np.roll(pts, -1, 0))
    path = np.vstack([pts, pts[:1]])
    nn = np.vstack([nrms, nrms[:1]])
    v, t, uv = ribbon(path, nn, WIDTH, THICK, segs=10, caps=False)
    parts = [Part("circlet", "CircletGold", v, t, uv, "circlet", "gold")]
    i = int(np.argmax(pts[:, 1]))
    fc, fn = pts[i], nrms[i]
    up = np.array([0, 0, 1.0])
    sv, st, suv = sun_disc(fc + fn * 0.12, fn, up, 0.85, 0.2, rays=14, ray_len=0.38, rings=4)
    parts.append(Part("circletsun", "CircletGold", sv, st, suv, "circletsun", "gold"))
    gv, gt, guv = dome(fc + fn * 0.3, fn, 0.34, 0.2)
    parts.append(Part("circletgem", "CircletGold", gv, gt, guv, "patch:obsidian", "gem"))
    down = mk_geom.normalize(np.array([0, 0, -1.0]) - fn * fn[2])
    dv, dt, duv = teardrop(fc + down * 1.55 + fn * 0.12, fn, down, 0.3, 0.95)
    parts.append(Part("circletdrop", "CircletGold", dv, dt, duv, "patch:gold", "gold"))
    gv, gt, guv = dome(fc + down * 1.6 + fn * 0.26, fn, 0.15, 0.09, seg=12, rings=3)
    parts.append(Part("circletdropgem", "CircletGold", gv, gt, guv, "patch:amber", "gem"))
    for p in parts:
        p.bones = ["NPC Head [Head]"]
        p.w = np.ones((len(p.v), 1))
        p.D = None
        p.sliders = []
    log("circlet: %d parts" % len(parts))
    return {"parts": parts, "bones": ["NPC Head [Head]"]}
