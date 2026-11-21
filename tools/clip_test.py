"""Clipping test of the built outfit (runs inside Blender for the ray casts).

  blender -b --factory-startup --python tools/clip_test.py -- <work> <out_data> <report_dir> [--quick]

Reads exactly what BodySlide reads (ShapeData NIF + OSD of every project), applies each test preset at weight 0
and 100 to the garments and to the skin (the trimmed UBE body of the Wrap, the full UBE body and feet for the other
pieces), and counts skin vertices that end up outside a garment that covers them in the reference shape ("pokes"),
with their depth. Then the same in animation poses (walk, run, sprint, sneak, sitting; HKX read by
tools/lib/hkx_anim.py) with linear blend skinning (the arms' twist bones posed with their parents,
see twist_parent). Garment-on-garment checks: the loincloth belt and panels over the briefs, the collar over the halter
straps, the cup fringe in front of the cups. The three slot-52 items (Loincloth with the briefs, LoinclothNoBriefs,
Briefs) are each tested on the whole body (the Wrap no longer trims the skin under the briefs).
The cloth on physics chains is left out of the pose test; tools/dev/veil_gravity.py hangs the arm veils under
gravity within their physics limits instead."""
import json
import os
import pickle
import re
import sys
import time

import numpy as np

TOOLS = os.path.dirname(os.path.abspath(__file__))
for sub in ("lib", "pieces"):
    sys.path.insert(0, os.path.join(TOOLS, sub))
import mk_blend  # noqa: E402
import mk_bs  # noqa: E402
import mk_fit  # noqa: E402
import mk_geom  # noqa: E402
import mk_nif  # noqa: E402
import mk_preset  # noqa: E402
import mk_testpresets  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1:]
WORK, DATA, OUT = argv[0], argv[1], argv[2]
QUICK = "--quick" in argv
SD = os.path.join(DATA, "CalienteTools", "BodySlide", "ShapeData", "Sun Dancer")
PRESETS = mk_testpresets.TEST_PRESETS     # tools/presets.local.json, else UBE's zeroed sliders
ANIMS = [("mt_walkforward", 6), ("mt_runforward", 4), ("mt_sprintforward", 3), ("sneak_idle", 2),
         ("chair_idle", 3)]
POKE_MIN = 0.05
t0 = time.time()


def log(m):
    print("[%6.1fs] %s" % (time.time() - t0, m), flush=True)


class Shape:
    def __init__(self, sh, diffs):
        self.name = sh["name"]
        self.v = sh["v"]
        self.t = sh["t"]
        self.bones = sh["bones"]
        self.w = sh["w"]
        self.diffs = diffs
        # partition (biped slot) of every triangle, when the shape has several
        ids, pt = sh.get("partitions") or [], sh.get("partition_tris") or []
        self.tri_slots = None
        if len(ids) > 1 and len(pt) == len(self.t):
            pt = np.asarray(pt)
            self.tri_slots = np.asarray(ids)[pt] if pt.max() < len(ids) else pt
        # weight on the cloth physics bones (SMP chains): that cloth is moved by the physics in game
        ev = [j for j, b in enumerate(self.bones) if b.startswith("EV ")]
        self.dyn = self.w[:, ev].sum(1) if ev else np.zeros(len(self.v))

    def static_tris(self):
        """Triangles not moved by the cloth physics (the pose test skins with the skeleton only)."""
        return self.t[~(self.dyn[self.t] > 0.5).all(1)]

    def morphed(self, vals):
        v = self.v.copy()
        for s, (idx, d) in self.diffs.items():
            x = vals.get(s, 0.0)
            if x:
                v[idx] += d * x
        return v


def load_project(stem):
    nif = mk_nif.NifFile(os.path.join(SD, stem + ".nif"))
    osd = mk_bs.read_osd(os.path.join(SD, stem + ".osd"))
    shapes = {}
    for s in nif.shapes:
        sh = mk_nif.read_shape(s)
        pre = sh["name"]
        diffs = {k[len(pre):]: v for k, v in osd.items() if k.startswith(pre)}
        # longest shape-name prefix wins (Wrap vs WrapGold / WrapLining)
        shapes[pre] = Shape(sh, diffs)
    for a in shapes:
        for b in shapes:
            if a != b and b.startswith(a):
                shapes[a].diffs = {k: v for k, v in shapes[a].diffs.items() if not k.startswith(b[len(a):])}
    return shapes


def skin_from_body(body, feet):
    v = body.v
    sh = {"name": "Body", "v": v, "t": body.t, "bones": body.bones, "w": body.w}
    diffs = {n: (np.arange(len(v)), body.D[:, k].astype(np.float64)) for k, n in enumerate(body.slider_names)}
    out = [Shape(sh, diffs)]
    if feet is not None:
        shf = {"name": "BodyFeet", "v": feet.v, "t": feet.t, "bones": feet.bones, "w": feet.w}
        dfs = {n: (np.arange(len(feet.v)), feet.D[:, k].astype(np.float64)) for k, n in enumerate(feet.slider_names)}
        out.append(Shape(shf, dfs))
    return out


def merge(meshes):
    vs, ts = [], []
    off = 0
    for v, t in meshes:
        vs.append(v)
        ts.append(t + off)
        off += len(v)
    return np.concatenate(vs), np.concatenate(ts)


def candidates(skin_v, skin_t, garment_v, garment_t, reach=1.6, parallel=False):
    """Skin vertices covered by the garment in the reference shape (garment found outward within reach).
    parallel (a garment over another garment): only where the outer layer's face lies the same way as the inner one
    at rest; the inner garment's rims (normals along the cloth) and the belt's lower rim resting on the briefs'
    waistband met the outer layer at a slant, and in a pose their rays slipped past it and counted as pokes."""
    n = mk_geom.vertex_normals(skin_v, skin_t)
    tree = mk_blend.bvh(garment_v, garment_t)
    _, gn, face, dist = mk_blend.raycast(tree, skin_v + n * 0.005, n, reach)
    ok = np.isfinite(dist)   # any face: closed pieces (gold bands) have their inner face in the same shape
    if parallel:
        fn = mk_geom.face_normals(garment_v, garment_t)
        fn = fn / np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
        dot = np.zeros(len(skin_v))
        dot[ok] = (fn[face[ok]] * n[ok]).sum(1)
        ok &= dot > 0.5
    return np.nonzero(ok)[0]


def pokes(skin_v, skin_t, idx, garment_v, garment_t, reach=2.5):
    if len(idx) == 0:
        return 0, 0.0, np.zeros(0, dtype=np.int64)
    n = mk_geom.vertex_normals(skin_v, skin_t)[idx]
    p = skin_v[idx]
    tree = mk_blend.bvh(garment_v, garment_t)
    _, gn, _, dout = mk_blend.raycast(tree, p + n * 0.005, n, reach)
    outside_ok = np.isfinite(dout)
    _, gn2, _, din = mk_blend.raycast(tree, p + n * 0.005, -n, reach)
    poke = ~outside_ok & np.isfinite(din) & ((gn2 * n).sum(1) > 0) & (din > POKE_MIN)
    depth = np.where(poke, din, 0.0)
    return int(poke.sum()), float(depth.max(initial=0.0)), idx[poke]


# ------------------------------------------------------------------ skinning
def qmat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def twist_parent(b):
    """The arms' twist bones are posed with their parent bone: the HKX reader decodes their tracks unreliably (in the
    walk, run and sprint animations the upper arm's twist bone came out turned 49 degrees about an axis across the
    arm in every frame, which would tear every UBE arm; in sneaking the same angle is a true twist about the arm).
    Skin and garments are posed the same way, so the test stays fair; only pure twist is not exercised."""
    for s in ("L", "R"):
        if b.startswith("NPC %s UpperarmTwist" % s):
            return "NPC %s UpperArm [%sUar]" % (s, s)
        if b.startswith("NPC %s ForearmTwist" % s):
            return "NPC %s Forearm [%sLar]" % (s, s)
    return b


def fallback_bone(b, index):
    b = twist_parent(b)
    if b in index:
        return b
    if b.startswith("EV Veil "):    # the veils' chains hang from the upper arms
        s = b.split()[2][0]
        return "NPC %s UpperArm [%sUar]" % (s, s)
    if b.startswith("EV "):    # the loincloth's chains hang from the pelvis at rest
        return "NPC Pelvis [Pelv]"
    side = " L " if " L " in b else (" R " if " R " in b else None)
    bl = b.lower()
    if "breast" in bl or "spine2" in bl or "clavicle" in bl:
        t = "NPC Spine2 [Spn2]"
    elif "belly" in bl or "spine1" in bl:
        t = "NPC Spine1 [Spn1]"
    elif any(k in bl for k in ("butt", "pelvis", "pussy", "vagina", "clitoral", "anus")):
        t = "NPC Pelvis [Pelv]"
    elif "thigh" in bl and side:
        t = "NPC %s Thigh [%sThg]" % (side.strip(), side.strip())
    elif "calf" in bl and side:
        t = "NPC %s Calf [%sClf]" % (side.strip(), side.strip())
    else:
        t = "NPC Spine [Spn0]"
    return t if t in index else "NPC Pelvis [Pelv]"


class Poser:
    def __init__(self, skel_hkx):
        import hkx_anim as H
        self.H = H
        self.sk = H.Skeleton(skel_hkx)
        self.ref = H.world_pose(self.sk, [], None)

    def pose(self, anim_path, frame_frac):
        an = self.H.Animation(anim_path)
        f = frame_frac * (an.nframes - 1)
        return self.H.world_pose(self.sk, an.pose(f), an.track_to_bone)

    def skin(self, v, bones, w, pw):
        out = np.zeros_like(v)
        tot = np.zeros(len(v))
        for j, b in enumerate(bones):
            wj = w[:, j]
            sel = wj > 0
            if not sel.any():
                continue
            bi = self.sk.index[fallback_bone(b, self.sk.index)]
            pp, pq = pw[bi]
            rp, rq = self.ref[bi]
            R = qmat(pq) @ qmat(rq).T
            t = np.array(pp) - R @ np.array(rp)
            out[sel] += wj[sel, None] * (v[sel] @ R.T + t)
            tot[sel] += wj[sel]
        tot[tot == 0] = 1
        return out / tot[:, None]


def main():
    os.makedirs(OUT, exist_ok=True)
    ref = pickle.load(open(os.path.join(WORK, "ref", "body.pkl"), "rb"))
    body = mk_fit.Body(ref)
    feet = mk_fit.Body(pickle.load(open(os.path.join(WORK, "ref", "feet.pkl"), "rb")), shape_names=("Feet",))
    presets = pickle.load(open(os.path.join(WORK, "ref", "presets.pkl"), "rb"))
    attrs = list(body.slider_attrs) + [a for a in feet.slider_attrs if a["name"] not in body.slider_names]
    configs = [(p, w) for p in PRESETS for w in ((0.0, 1.0) if not QUICK else (1.0,))]
    vals = {(p, w): mk_preset.slider_values(attrs, presets.get(p, {}), w) for p, w in configs}
    # extra bodies (e.g. the character's in-game body from the save: cache\extra_samples.json, not versioned)
    for name, v in mk_preset.extra_samples(WORK):
        configs.append((name, 1.0))
        vals[(name, 1.0)] = v
    full_body = skin_from_body(body, feet)
    projects = {s: load_project(s) for s in ("Wrap", "Loincloth", "LoinclothNoBriefs", "Briefs", "Sandals",
                                             "Bracelets", "Armlets", "Veils", "Collar", "Navel")}
    log("projects loaded")
    tests = []   # (name, skin shapes, garment shape names (project, shape))
    w = projects["Wrap"]
    tests.append(("Wrap vs skin", [w["BaseShape"]], [s for n, s in w.items() if n in ("Wrap", "WrapGold")]))
    # the skin as worn with the briefs: the Wrap's body without its partition under the briefs, which the game hides
    # while an item with the briefs (slot 55 too) is worn
    base = w["BaseShape"]
    worn = full_body[0]
    if base.tri_slots is not None and (base.tri_slots == 55).any():
        worn = Shape({"name": "BaseShape", "v": base.v, "t": base.t[base.tri_slots != 55], "bones": base.bones,
                      "w": base.w}, base.diffs)
        log("skin with the briefs on: %d of %d Wrap triangles (partition 55 hidden)" % (len(worn.t), len(base.t)))
    for stem in ("Loincloth", "LoinclothNoBriefs", "Briefs", "Bracelets", "Armlets", "Veils", "Collar", "Navel"):
        pr = projects[stem]
        gs = [s for n, s in pr.items() if not n.endswith("Lining") and not n.startswith("EV_")]
        tests.append(("%s vs body" % stem, [worn if stem in ("Loincloth", "Briefs") else full_body[0]], gs))
    sp = projects["Sandals"]
    tests.append(("Sandals vs body+feet", [full_body[0], sp["Feet"]],
                  [s for n, s in sp.items() if n in ("Sandals", "SandalsGold")]))
    # the loincloth's belt and panels over the briefs worn with it (the briefs are now their own shapes)
    lo = projects["Loincloth"]
    tests.append(("Loincloth vs briefs", [lo["Briefs"]], [lo["Loincloth"]]))
    tests.append(("Collar vs halter", [w["Wrap"]], [projects["Collar"]["CollarGold"]]))
    # the cup fringe (and the other gold on the Wrap) must stay in front of the cups and band: a bead that sinks
    # behind the cup's surface is a hidden fringe
    tests.append(("Fringe vs cups", [w["Wrap"]], [w["WrapGold"]]))
    results = {"static": {}, "poses": {}}
    cand_cache = {}
    for name, skins, garments in tests:
        sv0, st0 = merge([(s.v, s.t) for s in skins])
        gv0, gt0 = merge([(g.v, g.t) for g in garments])
        idx = candidates(sv0, st0, gv0, gt0, parallel=(name == "Loincloth vs briefs"))
        cand_cache[name] = idx
        rows = {}
        for (p, wgt) in configs:
            vv = vals[(p, wgt)]
            sv, st = merge([(s.morphed(vv), s.t) for s in skins])
            gv, gt = merge([(g.morphed(vv), g.t) for g in garments])
            n, dmax, which = pokes(sv, st, idx, gv, gt)
            rows["%s @%d" % (p, int(wgt * 100))] = {"pokes": n, "max_depth": round(dmax, 3),
                                                     "where": np.round(sv0[which][:12], 1).tolist()}
        results["static"][name] = {"covered_skin_verts": int(len(idx)), "presets": rows}
        worst = max(rows.values(), key=lambda r: r["pokes"])
        log("%-24s covered %5d | worst %d pokes (max depth %.2f)" % (name, len(idx), worst["pokes"],
                                                                       worst["max_depth"]))
    poser = Poser(os.path.join(WORK, "refs", "anims", "skeleton_female.hkx"))
    pose_presets = list(mk_testpresets.POSE_PRESETS) if not QUICK else mk_testpresets.POSE_PRESETS[:1]
    pose_presets += [name for name, _ in mk_preset.extra_samples(WORK)][:1]
    for anim, nfr in ANIMS:
        path = os.path.join(WORK, "refs", "anims", anim + ".hkx")
        if "--static" in argv:
            break
        for k in range(nfr):
            frac = (k + 0.5) / nfr
            try:
                pw = poser.pose(path, frac)
            except Exception as e:   # animation formats the reader does not handle
                log("pose %s skipped: %s" % (anim, e))
                break
            for p in pose_presets:
                vv = vals[(p, 1.0)]
                key = "%s %.2f | %s" % (anim, frac, p)
                row = {}
                for name, skins, garments in tests:
                    idx = cand_cache[name]
                    sv, st = merge([(poser.skin(s.morphed(vv), s.bones, s.w, pw), s.t) for s in skins])
                    # the cloth on physics chains is left out (in game the physics moves it out of the way)
                    gv, gt = merge([(poser.skin(g.morphed(vv), g.bones, g.w, pw), g.static_tris())
                                    for g in garments])
                    n, dmax, which = pokes(sv, st, idx, gv, gt)
                    sv_ref, _ = merge([(s.morphed(vv), s.t) for s in skins])
                    row[name] = {"pokes": n, "max_depth": round(dmax, 3),
                                 "where": np.round(sv_ref[which][:8], 1).tolist()}
                results["poses"][key] = row
                log("pose %s: %s" % (key, ", ".join("%s %d/%.2f" % (n.split(" vs")[0], r["pokes"], r["max_depth"])
                                                    for n, r in row.items())))
    json.dump(results, open(os.path.join(OUT, "clip_report.json"), "w"), indent=1)
    log("report written")


if __name__ == "__main__":
    main()
