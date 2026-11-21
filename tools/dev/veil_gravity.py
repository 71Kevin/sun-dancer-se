"""Arm veils hanging under gravity in animation poses (runs inside Blender for the nearest-point queries).

  blender -b --factory-startup --python tools/dev/veil_gravity.py -- <work> <out_data> <report_dir>
          [--limits y,lo_z,hi_z] [--presets a;b] [--render prefix] [--quick]

The clip test skins the veils' physics chains rigidly with the upper arm (as at rest). In game FSMP lets each chain
bone swing about the previous one, within the angular limits of the system file (veils.xml), and gravity pulls it
down. This test takes the worst case: every dynamic chain bone turned as far toward straight down as its limits allow
(the springs lose, nothing collides), then skins the veil with those chain bones and counts veil vertices that end up
inside the posed body (arms and torso), with their depth.

Limits follow Bullet's generic 6DOF constraint as FSMP builds it (bodyA = the chain bone, bodyB = the one above it,
frames at bodyB's origin): the measured angles are the Euler XYZ angles of frameA^-1 * frameB, i.e. of the inverse
of the child's turn in the parent's frame. A chain bone's local X runs down the chain and Y is the cloth's outward
normal, so a positive measured Z angle swings the child toward -Y, toward the arm.
Animations: the cached base game set plus the player's Gulan0 idles and walks (MO2 mods folder)."""
import glob
import json
import os
import re
import sys
import time

import numpy as np

TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("lib", "pieces", ""):
    sys.path.insert(0, os.path.join(TOOLS, sub))
argv = sys.argv[sys.argv.index("--") + 1:]
OPTS = {}
for key in ("--limits", "--presets", "--render", "--collide"):
    if key in argv:
        i = argv.index(key)
        OPTS[key] = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
QUICK = "--quick" in argv
argv = [a for a in argv if a != "--quick"]
WORK, DATA, OUT = argv[0], argv[1], argv[2]
sys.argv = [sys.argv[0], "--", WORK, DATA, OUT]
import clip_test as ct  # noqa: E402
import mk_blend  # noqa: E402
import mk_fit  # noqa: E402
import mk_nif  # noqa: E402
import mk_preset  # noqa: E402
import mk_testpresets  # noqa: E402

MODS = os.environ.get("MK_MODS", r"C:\Modding\TrueNorth\mods")
OAR = r"meshes\actors\character\animations\OpenAnimationReplacer"
INSIDE = 0.05
t0 = time.time()


def log(m):
    print("[%6.1fs] %s" % (time.time() - t0, m), flush=True)


def animations():
    """The player's own animations (OAR, in the MO2 mods folder): Gulan0's static idles, walks, runs and sprints
    (light armour), Vanargand's light-armour idle; and, for reference, the base game's walk, run and sprint (which the
    player never uses; on large bodies they press the arms into the torso), sneaking and sitting."""
    cached = os.path.join(WORK, "refs", "anims")
    out = [(os.path.join(cached, a + ".hkx"), n) for a, n in
           (("mt_walkforward", 4), ("mt_runforward", 4), ("mt_sprintforward", 3), ("sneak_idle", 2),
            ("chair_idle", 2))]

    def mod(name, *parts):
        return os.path.join(MODS, name, OAR, *parts)
    idles = sorted(glob.glob(os.path.join(MODS, "[[]NoDelete[]] Gulan0 - Idles (OAR, static poses)", OAR, "**",
                                          "*.hkx"), recursive=True))
    out += [(p, 2) for p in idles]
    out.append((mod("Vanargand Animations II - Female Idle Walk And Run cata", "CATA", "Light Armor", "female",
                    "mt_idle.hkx"), 3))
    gm = "[NoDelete] Gulan0 - Movement (OAR - base and alternatives)"
    walks = sorted(glob.glob(os.path.join(MODS, "[[]NoDelete[]] Gulan0 - Movement (OAR - base and alternatives)", OAR,
                                          "**", "mt_walkforward.hkx"), recursive=True))
    out += [(p, 3) for p in walks[:3]]
    cata = "[NoDelete] Gulan0 - Female Movement (CATA)"
    out += [(mod(cata, "CATA", "Light Armor", "female", "mt_runforward.hkx"), 4),
            (mod(cata, "CATA", "Light Armor", "female", "mt_sprintforward.hkx"), 3),
            (mod(gm, "Gulan0_walk_run_sprint", "Gulan0_walk run sprint Submod 1", "female", "mt_runforward.hkx"), 4),
            (mod(gm, "Gulan0 - Alternativas", "Sprint revisado (rework)", "female", "mt_sprintforward.hkx"), 3)]
    return [(p, n) for p, n in out if os.path.exists(p)]


def arm_in_torso(bv, bt, lab):
    """Arm skin inside the torso in this pose (the animation itself presses the arm into the body)."""
    arm_v = np.isin(lab, ("upperarm", "forearm"))
    tri_arm = arm_v[bt].any(1)
    tree = mk_blend.bvh(bv, bt[~tri_arm])
    p = bv[arm_v]
    loc, nrm, _, dist = mk_blend.nearest(tree, p)
    depth = -((p - loc) * nrm).sum(1)
    hit = np.isfinite(dist) & (dist < 4.0) & (depth > 0.3)
    return int(hit.sum()), float(np.where(hit, depth, 0).max(initial=0))


def read_limits(path):
    xml = open(path, encoding="utf-8").read()
    block = xml[xml.index("<generic-constraint-default>"):xml.index("</generic-constraint-default>")]

    def vec(tag):
        m = re.search(r'<%s x="([-\d.e]+)" y="([-\d.e]+)" z="([-\d.e]+)"/>' % tag, block)
        return np.array([float(m.group(k)) for k in (1, 2, 3)])
    return vec("angularLowerLimit"), vec("angularUpperLimit")


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def child_turns(lo, hi, ny=13, nz=41):
    """Every allowed turn of a chain bone in its parent's frame: Q = (Ry(y) Rz(z))^-1 for measured (y, z) inside
    the limits (twist left at 0: it does not move the bone's origin)."""
    out = []
    for y in np.linspace(lo[1], hi[1], ny):
        for z in np.linspace(lo[2], hi[2], nz):
            out.append(rot_z(-z) @ rot_y(-y))
    return np.array(out)


def transform(R, t):
    m = np.eye(4)
    m[:3, :3] = R
    m[:3, 3] = t
    return m


def hang_chains(chains, bind, T_parent, turns, gravity=np.array([0, 0, -1.0]), outside=None, spheres=None):
    """World matrices of every chain bone: anchors follow their parent bone, each dynamic bone turned (within the
    limits) to point its segment as far down as it can. outside(points) -> bool: with it, a turn that would put the
    bone's collision spheres (spheres[bone]: the EV_PhysCloth vertices it carries, bind space) or its segment's end
    inside the body is skipped for the next lowest one, as the physics collision stops the cloth there."""
    W = {}
    for names in chains:
        side = names[0].split()[2][0]
        W[names[0]] = T_parent[side] @ bind[names[0]]
        for k in range(1, len(names)):
            parent, child = names[k - 1], names[k]
            L = np.linalg.inv(bind[parent]) @ bind[child]
            seg = L[:3, 3]
            Wp = W[parent]
            dirs = np.einsum("ij,njk,k->ni", Wp[:3, :3], turns, seg)
            order = np.argsort(-(dirs @ gravity))
            pick = order[0]
            if outside is not None:
                local = np.linalg.inv(bind[child])
                pts = spheres.get(child, np.zeros((0, 3))) if spheres else np.zeros((0, 3))
                pl = pts @ local[:3, :3].T + local[:3, 3]
                for start in range(0, len(order), 24):
                    cand = order[start:start + 24]
                    found = None
                    for i in cand:
                        Wc = Wp @ transform(turns[i], np.zeros(3)) @ L
                        test = np.vstack([Wc[:3, 3][None], pl @ Wc[:3, :3].T + Wc[:3, 3]])
                        if outside(test).all():
                            found = i
                            break
                    if found is not None:
                        pick = found
                        break
            W[child] = Wp @ transform(turns[pick], np.zeros(3)) @ L
    return W


def skin(poser, pw, v, bones, w, chain_world, bind):
    out = np.zeros_like(v)
    tot = np.zeros(len(v))
    for j, b in enumerate(bones):
        wj = w[:, j]
        sel = wj > 0
        if not sel.any():
            continue
        if b in chain_world:
            S = chain_world[b] @ np.linalg.inv(bind[b])
            R, t = S[:3, :3], S[:3, 3]
        else:
            bi = poser.sk.index[ct.fallback_bone(b, poser.sk.index)]
            pp, pq = pw[bi]
            rp, rq = poser.ref[bi]
            R = ct.qmat(pq) @ ct.qmat(rq).T
            t = np.array(pp) - R @ np.array(rp)
        out[sel] += wj[sel, None] * (v[sel] @ R.T + t)
        tot[sel] += wj[sel]
    tot[tot == 0] = 1
    return out / tot[:, None]


def parent_transforms(poser, pw):
    out = {}
    for s in ("L", "R"):
        bi = poser.sk.index["NPC %s UpperArm [%sUar]" % (s, s)]
        pp, pq = pw[bi]
        rp, rq = poser.ref[bi]
        R = ct.qmat(pq) @ ct.qmat(rq).T
        out[s] = transform(R, np.array(pp) - R @ np.array(rp))
    return out


def inside(body_v, body_t, pts):
    tree = mk_blend.bvh(body_v, body_t)
    loc, nrm, face, dist = mk_blend.nearest(tree, pts)
    depth = -((pts - loc) * nrm).sum(1)
    hit = np.isfinite(dist) & (dist < 4.0) & (depth > INSIDE)
    return hit, np.where(hit, depth, 0.0), face


GROUPS = (("upperarm", ("UpperArm", "Upperarm")), ("forearm", ("Forearm", "Hand")),
          ("hip", ("Pelvis", "Thigh", "Butt", "Calf")), ("torso", ("Spine", "Breast", "Belly", "Clavicle")))


def body_groups(body):
    dom = np.argmax(body.w, 1)
    lab = np.full(len(body.v), "other", dtype=object)
    for j, b in enumerate(body.bones):
        for g, keys in GROUPS:
            if any(k in b for k in keys):
                lab[dom == j] = g
                break
    return lab


def where(hit, face, body_t, lab, cols=33, rows=None):
    """Hits by body region and by veil region (columns: front edge / middle / back edge; rows: top / middle / hem)."""
    out = {}
    if not hit.any():
        return out
    g = lab[body_t[face[hit], 0]]
    for name in np.unique(g):
        out["body:" + name] = int((g == name).sum())
    n = int(len(hit) // 2)
    idx = np.nonzero(hit)[0] % n
    rr, cc = idx // cols, idx % cols
    nrows = n // cols
    for name, sel in (("cols:edge0", cc < cols / 4), ("cols:mid", (cc >= cols / 4) & (cc < 3 * cols / 4)),
                      ("cols:edge1", cc >= 3 * cols / 4), ("rows:top", rr < nrows / 3),
                      ("rows:mid", (rr >= nrows / 3) & (rr < 2 * nrows / 3)), ("rows:hem", rr >= 2 * nrows / 3)):
        out[name] = int(sel.sum())
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    nif_path = os.path.join(ct.SD, "Veils.nif")
    nif = mk_nif.NifFile(nif_path)
    raw = {s.name: mk_nif.read_shape(s) for s in nif.shapes}
    shapes = ct.load_project("Veils")
    veil = shapes["Veils"]
    cloth = shapes["EV_PhysCloth"]
    # chain bones' rest transforms: the veil itself no longer carries the anchors ("... 00", counted as the upper arm
    # since round 5); the collider does
    bind = {}
    for sname in ("Veils", "EV_PhysCloth"):
        for b, m in zip(raw[sname]["bones"], raw[sname]["s2b"]):
            if b.startswith("EV ") and b not in bind:
                bind[b] = np.linalg.inv(m)
    names = sorted(bind)
    chains = []
    for n in names:
        if n.endswith(" 00"):
            chains.append([m for m in names if m.rsplit(" ", 1)[0] == n.rsplit(" ", 1)[0]])
    xml = os.path.join(DATA, "meshes", "!UBE", "Tinesh", "SunDancer", "veils.xml")
    lo, hi = read_limits(xml)
    if "--limits" in OPTS:
        y, lz, hz = (float(x) for x in OPTS["--limits"].split(","))
        lo, hi = np.array([lo[0], -y, lz]), np.array([hi[0], y, hz])
    turns = child_turns(lo, hi)
    log("veil %d verts, %d chains of %d bones; limits y %.2f..%.2f, z %.2f..%.2f (positive z = toward the arm)"
        % (len(veil.v), len(chains), len(chains[0]), lo[1], hi[1], lo[2], hi[2]))
    import pickle
    body = mk_fit.Body(pickle.load(open(os.path.join(WORK, "ref", "body.pkl"), "rb")))
    presets = pickle.load(open(os.path.join(WORK, "ref", "presets.pkl"), "rb"))
    attrs = list(body.slider_attrs)
    want = OPTS.get("--presets", ";".join(mk_testpresets.POSE_PRESETS)).split(";")
    configs = [(p, mk_preset.slider_values(attrs, presets.get(p, {}), 1.0)) for p in want if p in presets]
    extra = list(mk_preset.extra_samples(WORK))[:1]
    configs += extra
    body_shape = ct.skin_from_body(body, None)[0]
    lab = body_groups(body)
    poser = ct.Poser(os.path.join(WORK, "refs", "anims", "skeleton_female.hkx"))
    results = {"limits": {"lower": lo.tolist(), "upper": hi.tolist()}, "poses": {}}
    worst = {}
    for path, nfr in animations():
        tag = os.path.basename(os.path.dirname(os.path.dirname(path))) + "/" + os.path.basename(path)[:-4] \
            if "OpenAnimationReplacer" in path else os.path.basename(path)[:-4]
        for k in range(nfr if not QUICK else 1):
            frac = (k + 0.5) / nfr
            try:
                pw = poser.pose(path, frac)
            except Exception as e:
                log("%s skipped: %s" % (tag, e))
                break
            Tp = parent_transforms(poser, pw)
            cw = hang_chains(chains, bind, Tp, turns) if "--collide" not in OPTS else None
            for pname, vals in configs:
                bv = poser.skin(body_shape.morphed(vals), body_shape.bones, body_shape.w, pw)
                if "--collide" in OPTS:
                    margin = float(OPTS["--collide"])
                    tree = mk_blend.bvh(bv, body.t)

                    def outside(p, tree=tree, margin=margin):
                        loc, nrm, _, dist = mk_blend.nearest(tree, p)
                        return ~(np.isfinite(dist) & (dist < 6.0) & (((p - loc) * nrm).sum(1) < margin))
                    cv = cloth.morphed(vals)
                    spheres = {b: cv[(cloth.w[:, j] > 0.4)] for j, b in enumerate(cloth.bones) if b in bind}
                    cw = hang_chains(chains, bind, Tp, turns, outside=outside, spheres=spheres)
                vv = skin(poser, pw, veil.morphed(vals), veil.bones, veil.w, cw, bind)
                hit, depth, face = inside(bv, body.t, vv)
                key = "%s %.2f | %s" % (tag, frac, pname)
                ait = arm_in_torso(bv, body.t, lab)
                results["poses"][key] = {"inside": int(hit.sum()), "max_depth": round(float(depth.max()), 3),
                                         "left": int((hit & (veil.v[:, 0] < 0)).sum()),
                                         "right": int((hit & (veil.v[:, 0] > 0)).sum()),
                                         "where": where(hit, face, body.t, lab),
                                         "arm_in_torso": {"verts": ait[0], "max_depth": round(ait[1], 2)}}
                w0 = worst.get(tag, (0, 0.0))
                worst[tag] = (max(w0[0], int(hit.sum())), max(w0[1], float(depth.max())))
                if "--render" in OPTS and pname == configs[-1][0] and k == 0:
                    render(OPTS["--render"] + "_" + re.sub(r"[^A-Za-z0-9]+", "_", tag), bv, body.t, vv, veil.t, hit)
            log("%-46s %s" % ("%s %.2f" % (tag, frac), "  ".join(
                "%s %d/%.2f (arm in torso %d)" % (
                    p[:12], results["poses"]["%s %.2f | %s" % (tag, frac, p)]["inside"],
                    results["poses"]["%s %.2f | %s" % (tag, frac, p)]["max_depth"],
                    results["poses"]["%s %.2f | %s" % (tag, frac, p)]["arm_in_torso"]["verts"]) for p, _ in configs)))
    results["worst"] = {k: {"inside": a, "max_depth": round(b, 3)} for k, (a, b) in worst.items()}
    name = "veil_gravity_%s%s.json" % (OPTS.get("--limits", "xml").replace(",", "_"),
                                       "_collide" if "--collide" in OPTS else "")
    json.dump(results, open(os.path.join(OUT, name), "w"), indent=1)
    total = sum(a for a, _ in worst.values())
    log("worst per animation: %s" % ", ".join("%s %d/%.2f" % (k.split("/")[-1], a, b) for k, (a, b) in worst.items()))
    agg = {}
    for r in results["poses"].values():
        for k, n in r.get("where", {}).items():
            agg[k] = agg.get(k, 0) + n
    log("hits by region (all poses): %s" % ", ".join("%s %d" % kv for kv in sorted(agg.items())))
    log("TOTAL inside (worst frames summed): %d -> %s" % (total, name))


def render(prefix, bv, bt, vv, vt, hit):
    import bpy
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob)
    sk = mk_blend.mesh_object("body", bv, bt)
    sk.color = (0.75, 0.62, 0.55, 1)
    ve = mk_blend.mesh_object("veil", vv, vt)
    ve.color = (0.95, 0.75, 0.3, 1)
    if hit.any():
        # a small red octahedron on every veil vertex inside the body
        oct_v = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]) * 0.18
        oct_t = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]])
        pts = vv[hit]
        mv = (pts[:, None] + oct_v[None]).reshape(-1, 3)
        mt = (oct_t[None] + 6 * np.arange(len(pts))[:, None, None]).reshape(-1, 3)
        bad = mk_blend.mesh_object("hits", mv, mt)
        bad.color = (1, 0, 0, 1)
    mk_blend.setup_render(900, color_type="OBJECT")
    c = bv[:, :].mean(0)
    for view, d in (("back", (0, 1, 0)), ("right", (-1, 0, 0)), ("front", (0, -1, 0))):
        cam = mk_blend.camera((c[0], c[1], 85.0), d, 70)
        bpy.context.scene.render.filepath = "%s_%s.png" % (prefix, view)
        bpy.ops.render.render(write_still=True)
        bpy.data.objects.remove(cam)


main()
