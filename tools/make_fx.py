"""Effect palettes of the Eclipse Scythe's enchantment (Eclipse Brand):

  python tools/make_fx.py <work> <out_data> [<game data>]

The game's effect shaders colour grayscale effect textures through 2D palette gradients (256 x 64, the column is the
grayscale value, the row changes alpha and brightness). The enchantment's shaders keep the game's effect textures
and the structure of the base game's palettes (each texel's luminance and alpha), with the eclipse's own colour ramps
(black core, violet, amber, gold, white-gold): <out_data>/textures/Tinesh/SunDancer/fx/*.dds (BC3 with mips)."""
import os
import subprocess
import sys

import cv2
import numpy as np

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(TOOLS, "lib"))
TEXCONV = os.environ.get("MK_TEXCONV", r"C:\Modding\_tools\texconv\texconv.exe")
GAME = r"C:\Modding\TrueNorth\Stock Game\Data"
SRC = "textures\\effects\\gradients\\"
# output name: (base game palette, colour ramp as (luminance, rgb) stops)
PALETTES = {
    "GradEclipseCorona": ("gradflameench.dds", [(0.0, (0, 0, 0)), (0.12, (30, 8, 42)), (0.35, (140, 60, 18)),
                                                (0.6, (240, 150, 45)), (0.82, (255, 205, 110)),
                                                (1.0, (255, 248, 225))]),
    "GradEclipseFlare": ("gradfrosticeformorange.dds", [(0.0, (16, 6, 2)), (0.3, (150, 70, 18)),
                                                        (0.55, (240, 160, 50)), (0.8, (255, 215, 120)),
                                                        (1.0, (255, 252, 235))]),
    "GradEclipseSparks": ("gradfireexplosion.dds", [(0.0, (30, 14, 6)), (0.35, (200, 110, 30)),
                                                    (0.65, (255, 185, 70)), (1.0, (255, 250, 225))]),
    "GradEclipseUmbra": ("graddarkarmor.dds", [(0.0, (0, 0, 0)), (0.35, (14, 5, 22)), (0.6, (42, 16, 58)),
                                               (0.8, (120, 70, 30)), (0.92, (230, 165, 70)),
                                               (1.0, (255, 220, 150))]),
    "GradEclipseUmbraSmoke": ("graddisguiseshader02.dds", [(0.0, (0, 0, 0)), (0.5, (20, 8, 30)),
                                                           (0.8, (60, 30, 70)), (1.0, (180, 120, 60))]),
}


def find_in_bsas(game, names):
    found = {}
    for i in range(9):
        bsa = os.path.join(game, "Skyrim - Textures%d.bsa" % i)
        if not os.path.exists(bsa):
            continue
        out = subprocess.run([sys.executable, os.path.join(TOOLS, "lib", "bsa_get.py"), bsa, "--list", "gradients"],
                             capture_output=True, text=True).stdout.lower()
        for n in names:
            if n not in found and (SRC + n) in out:
                found[n] = bsa
    return found


def ramp(lum, stops):
    xs = np.array([s[0] for s in stops])
    out = np.zeros(lum.shape + (3,), np.float32)
    for c in range(3):
        out[..., c] = np.interp(lum, xs, [s[1][c] for s in stops])
    return out


def main(work, out_data, game=GAME):
    src_dir = os.path.join(work, "refs", "vanilla")
    names = sorted({v[0] for v in PALETTES.values()})
    missing = [n for n in names if not os.path.exists(os.path.join(src_dir, SRC + n))]
    if missing:
        for n, bsa in find_in_bsas(game, missing).items():
            subprocess.run([sys.executable, os.path.join(TOOLS, "lib", "bsa_get.py"), bsa, src_dir, SRC + n],
                           check=True, stdout=subprocess.DEVNULL)
    fx_dir = os.path.join(out_data, "textures", "Tinesh", "SunDancer", "fx")
    tmp = os.path.join(work, "fx")
    os.makedirs(fx_dir, exist_ok=True)
    os.makedirs(tmp, exist_ok=True)
    for name, (base, stops) in PALETTES.items():
        src = os.path.join(src_dir, SRC + base)
        subprocess.run([TEXCONV, "-nologo", "-y", "-ft", "png", "-f", "R8G8B8A8_UNORM", "-o", tmp, src], check=True,
                       stdout=subprocess.DEVNULL)
        img = cv2.imread(os.path.join(tmp, os.path.splitext(base)[0] + ".png"), cv2.IMREAD_UNCHANGED)
        rgb = img[..., [2, 1, 0]].astype(np.float32) / 255
        lum = np.clip(0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2], 0, 1)
        lum = lum / max(float(lum.max()), 1e-6)
        col = ramp(lum, stops)
        out = np.dstack([col[..., ::-1], img[..., 3].astype(np.float32)]).clip(0, 255).astype(np.uint8)
        png = os.path.join(tmp, name + ".png")
        cv2.imwrite(png, out)
        subprocess.run([TEXCONV, "-nologo", "-y", "-f", "BC3_UNORM", "-m", "0", "-o", fx_dir, png], check=True,
                       stdout=subprocess.DEVNULL)
        print("fx palette %s (from %s)" % (name, base))


if __name__ == "__main__":
    main(*sys.argv[1:])
