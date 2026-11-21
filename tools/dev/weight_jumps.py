"""Largest change of the bone weights across one edge, per shape, in the NIFs of a folder (runs inside Blender):
  blender -b --factory-startup --python tools/dev/weight_jumps.py -- <folder> [pattern, default *.nif]
A NIF keeps 4 bones per vertex; where neighbouring vertices keep different bones, a jump shows up here and, when
those bones move apart (physics chains, a crouch), a crease along the triangle edges in game. Even blends between
chains stay around 0.1 to 0.2 per edge; cut bones showed as 0.3 to 0.8 (round 5)."""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import mk_nif  # noqa: E402

args = sys.argv[sys.argv.index("--") + 1:]
folder = args[0]
pattern = args[1] if len(args) > 1 else "*.nif"
for path in sorted(glob.glob(os.path.join(glob.escape(folder), pattern))):
    nif = mk_nif.NifFile(path)
    for s in nif.shapes:
        if s.name.startswith("EV_"):
            continue
        sh = mk_nif.read_shape(s)
        v, t, w, bones = sh["v"], sh["t"], sh["w"], sh["bones"]
        if w is None or len(t) == 0:
            continue
        e = np.unique(np.sort(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), 1), axis=0)
        jump = 0.5 * np.abs(w[e[:, 0]] - w[e[:, 1]]).sum(1)
        i = int(np.argmax(jump))
        phys = any(b.startswith("EV ") for b in bones)
        print("%-26s %-18s v %6d bones %3d | jump max %.3f p99.9 %.3f n>0.1 %5d n>0.2 %4d | worst at (%.1f, %.1f, %.1f)%s"
              % (os.path.basename(path), s.name, len(v), len(bones), jump.max(), np.percentile(jump, 99.9),
                 (jump > 0.1).sum(), (jump > 0.2).sum(), *v[e[i, 0]], " physics" if phys else ""))
