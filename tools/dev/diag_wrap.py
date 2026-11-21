"""Diagnostics of the built Wrap against the full UBE body (runs inside Blender for the ray casts).

  blender -b --factory-startup --python tools/dev/diag_wrap.py -- <work> <out_data> <player_morphs.json>

For the zeroed body, two test presets and the player's in-game body (BodySlide base preset + OBody morphs read
from the save): (1) holes: skin triangles trimmed away under the Wrap that can be seen from outside (rays from the
trimmed skin escape past the garment and the kept skin), (2) dents: garment vertices that sink into the full body
(where the skin is trimmed this shows as a dent in the cups or briefs instead of a poke)."""
import json
import os
import pickle
import sys

import numpy as np

TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("lib", "pieces", ""):
    sys.path.insert(0, os.path.join(TOOLS, sub))
import mk_blend  # noqa: E402
import mk_fit  # noqa: E402
import mk_geom  # noqa: E402
import mk_preset  # noqa: E402
import mk_testpresets  # noqa: E402
from clip_test import load_project  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1:]
WORK, DATA, MORPHS = argv[0], argv[1], argv[2]


def player_values(path, weight):
    m = json.load(open(path))
    vals = {"SkinnyMorph": 1.0 - weight}
    for s, kv in m.items():
        if s.startswith("XPMSE") or s == "obody_processed":
            continue
        vals[s] = vals.get(s, 0.0) + sum(kv.values())
    return vals


def hemisphere(n, count=24):
    k = np.arange(count) + 0.5
    phi = np.arccos(1 - k / count * 0.92)          # up to ~85 degrees from the normal
    th = np.pi * (1 + 5 ** 0.5) * k
    loc = np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)
    a = np.where(np.abs(n[:, 2:3]) < 0.9, np.array([[0, 0, 1.0]]), np.array([[1.0, 0, 0]]))
    tx = mk_geom.normalize(np.cross(n, a))
    ty = np.cross(n, tx)
    return [(tx * d[0] + ty * d[1] + n * d[2], d[2]) for d in loc]


def main():
    ref = pickle.load(open(os.path.join(WORK, "ref", "body.pkl"), "rb"))
    body = mk_fit.Body(ref)
    presets = pickle.load(open(os.path.join(WORK, "ref", "presets.pkl"), "rb"))
    attrs = list(body.slider_attrs)
    sd = os.path.join(DATA, "CalienteTools", "BodySlide", "ShapeData", "Sun Dancer")
    import clip_test
    clip_test.SD = sd
    wrap = load_project("Wrap")
    trimmed = wrap["BaseShape"]
    key = lambda v: [tuple(r) for r in np.round(v / 1e-4).astype(np.int64)]   # noqa: E731
    pos = {k: i for i, k in enumerate(key(body.v))}
    kept_idx = np.array([pos.get(k, -1) for k in key(trimmed.v)])
    print("trimmed verts", len(trimmed.v), "matched", int((kept_idx >= 0).sum()))
    kept = np.zeros(len(body.v), dtype=bool)
    kept[kept_idx[kept_idx >= 0]] = True
    removed_t = body.t[~kept[body.t].all(1)]
    print("full body tris", len(body.t), "removed", len(removed_t))
    configs = [("Zeroed", {})]
    configs += [("%s @100" % mk_testpresets.label(p), mk_preset.slider_values(attrs, presets.get(p, {}), 1.0))
                for p in mk_testpresets.for_tool("diag_wrap", [])]
    configs += [("Tinesh in-game w0", player_values(MORPHS, 0.0)), ("Tinesh in-game w100", player_values(MORPHS, 1.0))]
    missing = [s for s in player_values(MORPHS, 0.0) if s not in body.slider_names]
    print("player sliders not in the body:", missing)
    gnames = [n for n in wrap if n != "BaseShape"]
    report = {}
    for name, vals in configs:
        bv = body.apply_preset(vals)
        tv = trimmed.morphed(vals)
        gs = [(wrap[n].morphed(vals), wrap[n].t) for n in gnames]
        occ_v = np.concatenate([tv] + [g[0] for g in gs])
        off = [0, len(tv)]
        for g in gs:
            off.append(off[-1] + len(g[0]))
        occ_t = np.concatenate([trimmed.t] + [g[1] + off[k + 1] for k, g in enumerate(gs)])
        tree = mk_blend.bvh(occ_v, occ_t)
        c = bv[removed_t].mean(1)
        n = mk_geom.face_normals(bv, removed_t)
        vis = np.zeros(len(c))
        tot = 0.0
        for d, w in hemisphere(n):
            _, _, _, dist = mk_blend.raycast(tree, c + n * 0.03, d, 80.0)
            vis += w * ~np.isfinite(dist)
            tot += w
        vis /= tot
        area = mk_geom.tri_area(bv, removed_t)
        bad = vis > 0.04
        holes = {"tris": int(bad.sum()), "area": round(float(area[bad].sum()), 2),
                 "worst_vis": round(float(vis.max(initial=0)), 3),
                 "where": np.round(c[bad][np.argsort(-vis[bad])][:10], 1).tolist()}
        # dents: garment (outer cloth) vertices below the full body surface
        btree = mk_blend.bvh(bv, body.t)
        gv = wrap["Wrap"].morphed(vals)
        loc, nrm, face, dist = mk_blend.nearest(btree, gv)
        sgn = np.sign(((gv - loc) * nrm).sum(1))
        sdist = sgn * dist
        regions = {"cups": gv[:, 2] > 80.5, "band": (gv[:, 2] > 87) & (gv[:, 2] < 92) & (np.abs(gv[:, 0]) < 3),
                   "briefs": gv[:, 2] < 75}
        dents = {}
        for rn, m in regions.items():
            inside = m & (sdist < 0.02)
            dents[rn] = {"verts": int(m.sum()), "below_skin": int(inside.sum()),
                         "min_sd": round(float(sdist[m].min(initial=9)), 3),
                         "where": np.round(gv[inside][np.argsort(sdist[inside])][:6], 1).tolist()}
        report[name] = {"holes": holes, "dents": dents}
        print("== %-22s holes %4d tris (area %.1f, worst vis %.2f) | dents %s" % (
            name, holes["tris"], holes["area"], holes["worst_vis"],
            ", ".join("%s %d/%.2f" % (k, v["below_skin"], v["min_sd"]) for k, v in dents.items())))
        print("   hole spots:", holes["where"][:6])
        for k, v in dents.items():
            if v["below_skin"]:
                print("   dent %s:" % k, v["where"][:4])
    out = os.path.join(WORK, "diag_wrap.json")
    json.dump(report, open(out, "w"), indent=1)
    print("written", out)


main()
