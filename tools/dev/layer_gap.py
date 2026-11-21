"""Clearance between two garment layers, every test preset and the in-game body (runs inside Blender):
  blender -b --factory-startup --python tools/dev/layer_gap.py -- <work> <out_data> <report_dir> [--stem Loincloth]
The clip test counts pokes deeper than 0.05 (below that, skin and garment differ by rounding). Two cloth layers that
nearly touch show in game too: where the gap closes, the inner layer pokes through in blocks along the outer one's
triangle edges. From every vertex of the briefs covered by the loincloth (belt and panels) a ray goes out along its
normal: the distance to the loincloth is the clearance; a loincloth face found inward instead is a poke (negative).
Physics cloth is measured in its rest shape."""
import json
import os
import pickle
import sys

import numpy as np

TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("lib", "pieces", ""):
    sys.path.insert(0, os.path.join(TOOLS, sub))
argv = sys.argv[sys.argv.index("--") + 1:]
STEM = argv[argv.index("--stem") + 1] if "--stem" in argv else "Loincloth"
argv = [a for a in argv if a not in ("--stem", STEM)]
sys.argv = [sys.argv[0], "--"] + argv[:3]
import clip_test as ct  # noqa: E402
import mk_blend  # noqa: E402
import mk_fit  # noqa: E402
import mk_geom  # noqa: E402
import mk_preset  # noqa: E402


def main():
    body = mk_fit.Body(pickle.load(open(os.path.join(ct.WORK, "ref", "body.pkl"), "rb")))
    presets = pickle.load(open(os.path.join(ct.WORK, "ref", "presets.pkl"), "rb"))
    configs = [(p, w) for p in ct.PRESETS for w in (0.0, 1.0)]
    vals = {(p, w): mk_preset.slider_values(body.slider_attrs, presets.get(p, {}), w) for p, w in configs}
    for name, v in mk_preset.extra_samples(ct.WORK):
        configs.append((name, 1.0))
        vals[(name, 1.0)] = v
    pr = ct.load_project(STEM)
    outer, inner = pr["Loincloth"], pr["Briefs"]
    # rest: the briefs' vertices the loincloth covers (found outward within 1.5)
    n0 = mk_geom.vertex_normals(inner.v, inner.t)
    _, _, _, d0 = mk_blend.raycast(mk_blend.bvh(outer.v, outer.t), inner.v + n0 * 0.005, n0, 1.5)
    cov = np.nonzero(np.isfinite(d0))[0]
    report = {"covered": int(len(cov)), "configs": {}}
    # below the belt (what shows under the panels) and under it (hidden by the belt), by the rest shape
    import piece_loincloth as L
    center = mk_fit.sym_curve(L.BELT_CENTER)
    th = np.arctan2(inner.v[cov, 0], inner.v[cov, 1])
    shown = inner.v[cov, 2] < center(th) - L.BELT_HALF - 0.1
    print("briefs vertices under the loincloth: %d (%d below the belt)" % (len(cov), shown.sum()), flush=True)
    for (p, w) in configs:
        vv = vals[(p, w)]
        gv, iv = outer.morphed(vv), inner.morphed(vv)
        n = mk_geom.vertex_normals(iv, inner.t)[cov]
        q = iv[cov]
        tree = mk_blend.bvh(gv, outer.t)
        _, gn, face, dout = mk_blend.raycast(tree, q + n * 0.005, n, 2.0)
        _, gn2, _, din = mk_blend.raycast(tree, q + n * 0.005, -n, 1.0)
        poke = ~np.isfinite(dout) & np.isfinite(din) & ((gn2 * n).sum(1) > 0)
        # only a face of the loincloth lying the same way as the briefs is a layer over them; the belt's lower rim
        # (facing down and in) rests on the waistband, a contact that shows nothing through
        fn = mk_geom.face_normals(gv, outer.t)
        fn = fn / np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
        hit = np.isfinite(dout)
        same = np.zeros(len(q), dtype=bool)
        same[hit] = (fn[face[hit]] * n[hit]).sum(1) > 0.3
        gap = np.where(hit & same, dout, np.nan)
        gap[poke] = -din[poke]
        g = gap[np.isfinite(gap)]
        gs = np.where(shown, gap, np.nan)
        s = gs[np.isfinite(gs)]
        row = {"pokes": int(poke.sum()), "deepest": round(float(-g.min()) if g.min() < 0 else 0.0, 3),
               "under_0.05": int((g < 0.05).sum()), "under_0.1": int((g < 0.1).sum()),
               "min": round(float(g.min()), 3), "where_min": np.round(q[np.nanargmin(gap)], 1).tolist(),
               "below_belt": {"under_0.05": int((s < 0.05).sum()), "under_0.1": int((s < 0.1).sum()),
                              "under_0.15": int((s < 0.15).sum()), "min": round(float(s.min()), 3),
                              "where_min": np.round(q[np.nanargmin(gs)], 1).tolist()}}
        report["configs"]["%s @%d" % (p, int(w * 100))] = row
        b = row["below_belt"]
        print("%-44s pokes %3d | all: < 0.05 %3d, min %+.3f | below the belt: < 0.1 %3d, < 0.15 %3d, min %+.3f at %s"
              % ("%s @%d" % (p, int(w * 100)), row["pokes"], row["under_0.05"], row["min"], b["under_0.1"],
                 b["under_0.15"], b["min"], b["where_min"]), flush=True)
    os.makedirs(ct.OUT, exist_ok=True)
    json.dump(report, open(os.path.join(ct.OUT, "layer_gap_%s.json" % STEM.lower()), "w"), indent=1)


main()
