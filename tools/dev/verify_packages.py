"""Checks the release packages in dist\\: archive integrity, layout, identical plugin/meshes/BodySlide data in the
three outfit packages, DDS sizes per resolution, PBR add-on contents, and no add-on sliders in the BodySlide data.

  python tools\\dev\\verify_packages.py [version]"""
import hashlib
import os
import re
import struct
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIST = os.path.join(ROOT, "dist")
SEVENZIP = os.environ.get("MK_7Z", r"C:\Modding\_tools\7z\7z.exe")
VERSION = sys.argv[1] if len(sys.argv) > 1 else "1.0"
SIZES = {"2K": 2048, "4K": 4096, "8K": 8192}
# the sliders of the add-on to the UBE slider sets, when it is installed: none may reach a package
ADDON = os.path.join(os.environ.get("MK_MODS", r"C:\Modding\TrueNorth\mods"), r"[NoDelete] UBE 2.0 - Necoco Body v2 (Discord)",
                     r"CalienteTools\Bodyslide\SliderSets")
problems = []


def bad(msg):
    problems.append(msg)
    print("PROBLEM:", msg)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dds_size(path):
    with open(path, "rb") as f:
        head = f.read(148)
    h, w = struct.unpack_from("<II", head, 12)
    mips = struct.unpack_from("<I", head, 28)[0]
    fourcc = head[84:88]
    dxgi = struct.unpack_from("<I", head, 128)[0] if fourcc == b"DX10" else None
    return w, h, mips, fourcc.decode("latin-1"), dxgi


def addon_slider_names():
    names = set()
    if not os.path.isdir(ADDON):
        return names
    from xml.etree import ElementTree as ET
    for f in os.listdir(ADDON):
        for s in ET.parse(os.path.join(ADDON, f)).getroot().iter("Slider"):
            if any("Necoco" in (d.text or "") for d in s.iter("Data")):
                names.add(s.get("name"))
    return names


def main():
    tmp = tempfile.mkdtemp(prefix="sd_pkg_")
    shared = {}
    addon = addon_slider_names()
    for res, px in SIZES.items():
        for kind in ("UBE", "PBR for PG Patcher"):
            name = "Sun Dancer - %s (%s) - %s.7z" % (kind, res, VERSION)
            arc = os.path.join(DIST, name)
            if not os.path.exists(arc):
                bad("missing " + name)
                continue
            t = subprocess.run([SEVENZIP, "t", arc], capture_output=True, text=True)
            if "Everything is Ok" not in t.stdout:
                bad("7z t failed: " + name)
            out = os.path.join(tmp, kind[:3] + res)
            subprocess.run([SEVENZIP, "x", "-y", "-o" + out, arc], capture_output=True, check=True)
            files = [os.path.relpath(os.path.join(d, f), out) for d, _, fs in os.walk(out) for f in fs]
            tops = sorted({f.split(os.sep)[0] for f in files})
            if kind == "UBE":
                want = ["CalienteTools", "[Tinesh] Sun Dancer.esp", "meshes", "textures"]
                if tops != want:
                    bad("%s: top level %s" % (name, tops))
                for f in files:
                    if not f.lower().startswith("textures" + os.sep + "tinesh" + os.sep + "sundancer" + os.sep) \
                            or os.sep + "fx" + os.sep in f:
                        shared.setdefault(f, {})[res] = sha(os.path.join(out, f))
                main_dds = os.path.join(out, "textures", "Tinesh", "SunDancer", "SunDancer.dds")
                n_dds = os.path.join(out, "textures", "Tinesh", "SunDancer", "SunDancer_n.dds")
                m_dds = os.path.join(out, "textures", "Tinesh", "SunDancer", "SunDancer_m.dds")
                for p, want_px in ((main_dds, px), (n_dds, px), (m_dds, px // 4)):
                    w, h, mips, cc, dxgi = dds_size(p)
                    full = int(round(__import__("math").log2(max(w, h)))) + 1
                    if (w, h) != (want_px, want_px) or mips != full:
                        bad("%s: %s is %dx%d with %d mips" % (name, os.path.basename(p), w, h, mips))
                osp = open(os.path.join(out, "CalienteTools", "BodySlide", "SliderSets", "Sun Dancer.osp"),
                           encoding="utf-8").read()
                sliders = {s.replace("&gt;", ">").replace("&lt;", "<").replace("&amp;", "&")
                           for s in re.findall(r'<Slider name="([^"]+)"', osp)}
                leaked = sliders & addon
                osd_text = b"".join(open(os.path.join(d, f), "rb").read() for d, _, fs in
                                    os.walk(os.path.join(out, "CalienteTools", "BodySlide", "ShapeData"))
                                    for f in fs if f.endswith(".osd"))
                if leaked or b"Necoco" in osd_text:
                    bad("%s: add-on sliders in the BodySlide data: %s" % (name, sorted(leaked)[:10]))
                print("%-45s ok: %d files, %d sliders, atlas %d px, %.1f MB"
                      % (name, len(files), len(sliders), px, os.path.getsize(arc) / 2 ** 20))
            else:
                if tops != ["PBRNifPatcher", "textures"]:
                    bad("%s: top level %s" % (name, tops))
                alb = os.path.join(out, "textures", "pbr", "Tinesh", "SunDancer", "SunDancer.dds")
                rmaos = os.path.join(out, "textures", "pbr", "Tinesh", "SunDancer", "SunDancer_rmaos.dds")
                for p, want_px in ((alb, px), (rmaos, px // 2)):
                    w, h, mips, cc, dxgi = dds_size(p)
                    if (w, h) != (want_px, want_px):
                        bad("%s: %s is %dx%d" % (name, os.path.basename(p), w, h))
                print("%-45s ok: %d files (%s), albedo %d px, %.1f MB"
                      % (name, len(files), ", ".join(sorted(os.path.basename(f) for f in files if f.endswith(".json"))),
                         px, os.path.getsize(arc) / 2 ** 20))
    differ = [f for f, h in shared.items() if len(set(h.values())) != 1 or len(h) != 3]
    print("files shared by the three outfit packages: %d, identical in all three: %d"
          % (len(shared), len(shared) - len(differ)))
    for f in differ[:10]:
        bad("differs between resolutions: " + f)
    print("PROBLEMS", len(problems))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
