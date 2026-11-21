import os
import re

import numpy as np

MODS = os.environ.get("MK_MODS", r"C:\Modding\TrueNorth\mods")


def read_preset(path, name=None):
    s = open(path, encoding="utf-8-sig", errors="replace").read()
    presets = re.findall(r'<Preset name="([^"]+)"[^>]*>(.*?)</Preset>', s, re.S)
    for pname, body in presets:
        if name is None or pname == name:
            vals = {}
            for m in re.finditer(r'<SetSlider name="([^"]+)" size="(small|big)" value="(-?[0-9.]+)"', body):
                name = m.group(1).replace("&gt;", ">").replace("&lt;", "<").replace("&amp;", "&")
                vals.setdefault(name, {})[m.group(2)] = float(m.group(3)) / 100.0
            return pname, vals
    raise KeyError(name)


def slider_values(slider_attrs, preset_vals, weight=1.0):
    out = {}
    for a in slider_attrs:
        n = a["name"]
        small = float(a.get("small", 0)) / 100.0
        big = float(a.get("big", 0)) / 100.0
        pv = preset_vals.get(n, {})
        small = pv.get("small", small)
        big = pv.get("big", big)
        val = small + (big - small) * weight
        if a.get("invert") == "true":
            val = 1.0 - val
        if a.get("uv") == "true":
            continue
        out[n] = val
    return out


def corpus(slider_attrs, mods_dir=MODS, weights=(0.0, 1.0), set_filter="UBE"):
    """Every BodySlide preset in the mods folder whose slider set name contains set_filter, at the given weights.
    Duplicate preset names (the same preset installed by several mods) are read once."""
    out = []
    seen = set()
    for root, dirs, files in os.walk(mods_dir):
        if os.path.basename(root).lower() != "sliderpresets":
            continue
        for fn in sorted(files):
            if not fn.lower().endswith(".xml"):
                continue
            path = os.path.join(root, fn)
            s = open(path, encoding="utf-8-sig", errors="replace").read()
            for pname, attrs in re.findall(r'<Preset name="([^"]+)"([^>]*)>', s):
                m = re.search(r'set="([^"]*)"', attrs)
                if set_filter and (not m or set_filter.lower() not in m.group(1).lower()):
                    continue
                if pname in seen:
                    continue
                seen.add(pname)
                _, vals = read_preset(path, pname)
                for wgt in weights:
                    out.append((pname, slider_values(slider_attrs, vals, wgt)))
    return out


def extra_samples(work):
    """Extra body samples for the fitting and the tests: <work>/extra_samples.json, a list of {"name": ..., "values":
    {slider: value}} (e.g. a character's in-game body: BodySlide base preset plus its RaceMenu/OBody morphs). The
    cache folder is not versioned, so personal bodies stay out of the repository."""
    import json
    path = os.path.join(work, "extra_samples.json")
    if not os.path.exists(path):
        return []
    return [(e["name"], {k: float(v) for k, v in e["values"].items()}) for e in json.load(open(path, encoding="utf-8"))]


def apply(v, D, sliders, values):
    out = np.array(v, dtype=np.float64)
    for j, n in enumerate(sliders):
        if n in values and values[n] != 0:
            out += D[:, j] * values[n]
    return out
