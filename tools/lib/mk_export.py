import os

import numpy as np

import mk_bs
import mk_geom
import mk_nif

TEX_DIR = "textures\\Tinesh\\SunDancer\\"
# texture sets: <stem>.dds diffuse (no _d suffix, so PG Patcher's PBR JSON can match the path), _n normal with
# specular in alpha, _m environment mask, _g glow
STEM = "SunDancer"
# sliders left out of the BodySlide data and the RaceMenu morphs: build_meshes fills it for the public packages with
# the sliders of add-ons to the UBE slider sets, whose data are not ours to redistribute
EXCLUDE_SLIDERS = set()


def texset(stem=STEM, env=None, glow=False):
    t = {"Diffuse": TEX_DIR + stem + ".dds", "Normal": TEX_DIR + stem + "_n.dds"}
    if env:
        t["EnvMap"] = env
        t["EnvMask"] = TEX_DIR + stem + "_m.dds"
    if glow:
        t["Glow"] = TEX_DIR + stem + "_g.dds"
    return t


BASE = {"Emissive_Color": [0.0, 0.0, 0.0, 0.0], "Emissive_Mult": 1.0, "Alpha": 1.0, "UV_Scale_U": 1.0,
        "UV_Scale_V": 1.0, "Env_Map_Scale": 1.0}


def shader(key, stem=STEM):
    if key == "cloth":
        return dict(BASE, Shader_Type=0, Shader_Flags_1=0x82400303, Shader_Flags_2=0x8001, Glossiness=38.0,
                    Spec_Str=1.0, Spec_Color=[1.0, 0.96, 0.9]), texset(stem)
    if key == "gold":
        return dict(BASE, Shader_Type=1, Shader_Flags_1=0x82400383, Shader_Flags_2=0x8001, Glossiness=78.0,
                    Spec_Str=1.7, Spec_Color=[1.0, 0.84, 0.52], Env_Map_Scale=1.0), \
            texset(stem, env="textures\\cubemaps\\bronze_e.dds")
    if key == "steel":
        return dict(BASE, Shader_Type=1, Shader_Flags_1=0x82400383, Shader_Flags_2=0x8001, Glossiness=82.0,
                    Spec_Str=1.9, Spec_Color=[0.92, 0.94, 1.0], Env_Map_Scale=1.0), \
            texset(stem, env="textures\\cubemaps\\ShinyDull_e.dds")
    if key == "glow":
        return dict(BASE, Shader_Type=2, Shader_Flags_1=0x82400303, Shader_Flags_2=0x8041, Glossiness=40.0,
                    Spec_Str=0.8, Spec_Color=[1.0, 0.9, 0.7], Emissive_Color=[1.0, 0.72, 0.3, 1.0],
                    Emissive_Mult=1.8), texset(stem, glow=True)
    if key == "chiffon":
        # a single sheet seen from both sides, blended by the texture's alpha (the fabric's sheerness): no shadow
        # casting and no depth writes (a transparent veil must not hide what is behind it)
        return dict(BASE, Shader_Type=0, Shader_Flags_1=0x82400103, Shader_Flags_2=0x8010, Glossiness=24.0,
                    Spec_Str=0.6, Spec_Color=[1.0, 0.9, 0.75]), texset(stem)
    raise KeyError(key)


SHADERS = {k: shader(k) for k in ("cloth", "gold", "steel", "glow", "chiffon")}
CHIFFON_ALPHA = (0x00ED, 0)    # alpha blending (source alpha, one minus source alpha)

# body sliders a garment does not follow (the skin under the garment is trimmed or covered)
NO_GARMENT = ("Nipple", "Areola", "Labia", "Vagina", "Clit", "Anal", "Anus", "Pussy", "Penis", "Innie",
              "Cutepuffyness", "CBPC", "Normals_recalc", "Nyl2")
SKINNED = 0x2
SMP_BODY_BONES = ("NPC Pelvis [Pelv]", "NPC Spine [Spn0]", "NPC Spine1 [Spn1]", "NPC Spine2 [Spn2]",
                  "NPC L Thigh [LThg]", "NPC R Thigh [RThg]", "NPC L Calf [LClf]", "NPC R Calf [RClf]",
                  "NPC L Butt", "NPC R Butt", "NPC L RearThigh", "NPC R RearThigh", "NPC L FrontThigh",
                  "NPC R FrontThigh", "NPC L RearCalf [LrClf]", "NPC R RearCalf [RrClf]", "NPC Belly")


def smp_bone_for(name):
    if name in SMP_BODY_BONES:
        return name
    if any(k in name for k in ("Breast", "Clavicle", "Upperarm", "UpperArm", "Neck")):
        return "NPC Spine2 [Spn2]"
    for s in ("L", "R"):
        if "NPC %s Foot" % s in name or "NPC %s Toe" % s in name:
            return "NPC %s Calf [%sClf]" % (s, s)
    return "NPC Pelvis [Pelv]"


def collapse_to_smp_bones(w, bones):
    out = list(SMP_BODY_BONES)
    wn = np.zeros((len(w), len(out)))
    for j, b in enumerate(bones):
        wn[:, out.index(smp_bone_for(b))] += w[:, j]
    wn /= np.maximum(wn.sum(1, keepdims=True), 1e-9)
    return wn, out


def merge_parts(parts, bone_order=None):
    bones = list(bone_order or [])
    sliders = []
    for p in parts:
        for b in p.bones:
            if b not in bones:
                bones.append(b)
        for s in p.sliders:
            if s not in sliders:
                sliders.append(s)
    vs, ts, uvs, ns = [], [], [], []
    W = []
    D = []
    off = 0
    for p in parts:
        n = len(p.v)
        vs.append(p.v)
        ts.append(p.t + off)
        uvs.append(p.uv_final)
        ns.append(p.normals if getattr(p, "normals", None) is not None else mk_geom.vertex_normals(p.v, p.t))
        w = np.zeros((n, len(bones)))
        for j, b in enumerate(p.bones):
            w[:, bones.index(b)] += p.w[:, j]
        W.append(w)
        d = np.zeros((n, len(sliders), 3), dtype=np.float32)
        if p.D is not None and len(p.sliders):
            for j, s in enumerate(p.sliders):
                if any(k in s for k in NO_GARMENT) and not getattr(p, "keep_all_sliders", False):
                    continue
                d[:, sliders.index(s)] = p.D[:, j]
        D.append(d)
        off += n
    return {"v": np.concatenate(vs), "t": np.concatenate(ts), "uv": np.concatenate(uvs), "n": np.concatenate(ns),
            "bones": bones, "w": np.concatenate(W), "sliders": sliders, "D": np.concatenate(D)}


def sparse_diffs(D, sliders, tol=2e-4, min_peak=2e-3):
    out = {}
    for j, s in enumerate(sliders):
        d = D[:, j]
        mag = np.abs(d).max(1)
        if mag.max(initial=0) < min_peak:
            continue
        idx = np.nonzero(mag > tol)[0]
        if len(idx):
            out[s] = (idx, d[idx].astype(np.float64))
    return out


class NifSet:
    def __init__(self, stem, out_name, partition, bone_globals):
        self.stem = stem
        self.out_name = out_name
        self.partition = partition
        self.bone_globals = bone_globals
        self.shapes = []
        self.root_extras = []
        self.nodes = []

    def add_shape(self, name, data, shader_key=None, shader=None, textures=None, extras=(), alpha=None,
                  partition=None, keep_all=False, tri_parts=None):
        if shader_key == "invisible":
            sh, tex = INVISIBLE
            alpha = INVISIBLE_ALPHA
        elif shader_key is not None:
            sh, tex = SHADERS[shader_key]
            if shader_key == "chiffon":
                alpha = CHIFFON_ALPHA
        else:
            sh, tex = shader, textures
        if len(data["v"]) > 65535 or len(data["t"]) > 65535:
            raise ValueError("shape %s exceeds BSTriShape limits: %d verts, %d tris" % (name, len(data["v"]),
                                                                                      len(data["t"])))
        diffs = sparse_diffs(data["D"], data["sliders"], *(() if not keep_all else (0.0, 0.0)))
        diffs = {k: v for k, v in diffs.items() if k not in EXCLUDE_SLIDERS}
        self.shapes.append({"name": name, "data": data, "shader": dict(sh), "textures": dict(tex),
                            "extras": list(extras), "alpha": alpha, "diffs": diffs, "tri_parts": tri_parts,
                            "partition": partition if partition is not None else self.partition})

    def _write_nif(self, path, extras_extra=None, with_bodytri=None):
        nif = mk_nif.new_nif(path)
        for name, value in self.root_extras:
            mk_nif.add_root_string(nif, name, value)
        if self.nodes:
            write_nodes(nif, self.nodes)
        for k, s in enumerate(self.shapes):
            d = s["data"]
            extras = list(s["extras"])
            if with_bodytri and k == 0:
                extras.append(("BODYTRI", with_bodytri))
            mk_nif.add_skinned_shape(nif, s["name"], d["v"], d["t"], d["uv"], d["n"], d["bones"], d["w"],
                                     self.bone_globals, s["partition"], s["shader"], s["textures"],
                                     alpha=s["alpha"], extras=extras, tri_parts=s["tri_parts"])
        nif.save()

    def write_single(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._write_nif(path)

    def write_shapedata(self, folder):
        os.makedirs(folder, exist_ok=True)
        self._write_nif(os.path.join(folder, self.stem + ".nif"))
        osd = {}
        for s in self.shapes:
            for slider, (idx, d) in s["diffs"].items():
                osd[s["name"] + slider] = (idx, d)
        mk_bs.write_osd(os.path.join(folder, self.stem + ".osd"), osd)

    def write_meshes(self, folder, tri_rel, slider_attrs):
        os.makedirs(folder, exist_ok=True)
        uv_sliders = {a["name"] for a in slider_attrs if a.get("uv") == "true"}
        for suffix in ("_0", "_1"):
            self._write_nif(os.path.join(folder, self.out_name + suffix + ".nif"), with_bodytri=tri_rel)
        shapes = []
        for s in self.shapes:
            morphs = [(k, v) for k, v in s["diffs"].items() if k not in uv_sliders]
            shapes.append((s["name"], morphs))
        mk_bs.write_tri(os.path.join(folder, self.out_name + ".tri"), shapes)

    def slider_set(self, set_name, data_folder, output_path, slider_attrs):
        sliders = []
        for attrs in slider_attrs:
            data = []
            for s in self.shapes:
                if attrs["name"] in s["diffs"]:
                    key = s["name"] + attrs["name"]
                    data.append((key, s["name"], "%s.osd\\%s" % (self.stem, key)))
            if data:
                sliders.append((attrs, data))
        return mk_bs.slider_set_xml(set_name, data_folder, self.stem + ".nif", output_path, self.out_name,
                                    [s["name"] for s in self.shapes], sliders)


def body_shape_data(ref, body, keep_tris, shape="BaseShape", D_weld=None, changed=None):
    """The reference body shape with only the kept triangles, every slider (texture-coordinate sliders included),
    the skin near the garment taking the conformed slider data D_weld; also returns the per-triangle partitions."""
    sh = ref["shapes"][shape]
    t = sh["t"][keep_tris]
    used = np.unique(t)
    remap = -np.ones(len(sh["v"]), dtype=np.int64)
    remap[used] = np.arange(len(used))
    start = body.ranges[shape][0]
    weld_idx = body.inv[start + np.arange(len(sh["v"]))]
    uv_names = {a["name"] for a in body.uv_attrs}
    use_new = np.zeros(len(sh["v"]), dtype=bool)
    if changed is not None:
        mark = np.zeros(len(body.v), dtype=bool)
        mark[changed] = True
        use_new = mark[weld_idx]
    sliders = []
    cols = []
    for attrs, data in ref["sliders"]:
        for dname, tgt, _ in data:
            if tgt != shape:
                continue
            name = attrs["name"]
            idx, d = ref["diffs"][(name, tgt)]
            dense = np.zeros((len(sh["v"]), 3), dtype=np.float32)
            if len(idx):
                dense[idx] = d
            if D_weld is not None and name not in uv_names and name in body.slider_names:
                k = body.slider_names.index(name)
                dense[use_new] = D_weld[weld_idx[use_new], k]
            sliders.append(name)
            cols.append(dense[used])
    D = np.stack(cols, 1)
    n = mk_geom.vertex_normals(body.v, body.t)[weld_idx]
    parts = np.asarray(sh["partition_tris"])[keep_tris] if len(sh["partitions"]) > 1 else None
    return {"v": sh["v"][used], "t": remap[t], "uv": sh["uv"][used], "n": n[used], "bones": sh["bones"],
            "w": sh["w"][used], "sliders": sliders, "D": D, "partitions": list(sh["partitions"]),
            "tri_parts": parts}, sh


INVISIBLE = ({"Shader_Type": 0, "Shader_Flags_1": 0x80000000, "Shader_Flags_2": 0x0, "Glossiness": 1.0,
              "Spec_Str": 0.0, "Spec_Color": [0.0, 0.0, 0.0], "Emissive_Color": [0.0, 0.0, 0.0, 0.0],
              "Emissive_Mult": 0.0, "Alpha": 0.0, "UV_Scale_U": 1.0, "UV_Scale_V": 1.0, "Env_Map_Scale": 1.0}, texset())
INVISIBLE_ALPHA = (0x00ED, 0)


def write_nodes(nif, nodes):
    made = {}
    for name, parent, glob in nodes:
        pg = np.eye(4) if parent is None else dict((n, g) for n, _, g in nodes)[parent]
        local = np.linalg.inv(pg) @ glob
        made[name] = nif.add_node(name, mk_nif.mat_to_xf(local), made.get(parent))
    return made


def static_shapes(nif, groups, parent=None):
    out = []
    for name, data, shader_key, alpha in groups:
        sh, tex = SHADERS[shader_key] if isinstance(shader_key, str) else shader_key
        sh = dict(sh, Shader_Flags_1=sh["Shader_Flags_1"] & ~SKINNED)
        out.append(mk_nif.add_static_shape(nif, name, data["v"], data["t"], data["uv"], data["n"], sh,
                                           dict(tex), alpha=alpha, parent=parent))
    return out
