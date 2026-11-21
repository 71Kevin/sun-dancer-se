"""Are the bracelets rigid? Counts the distinct bone-weight vectors per wrist in a built NIF (one per wrist = the whole
set moves as one rigid piece) and lists the bones they use (runs in Blender):
  blender -b --factory-startup --python tools/dev/rigid_check.py -- <nif> [<nif> ...]"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import mk_nif  # noqa: E402

for path in sys.argv[sys.argv.index("--") + 1:]:
    for s in mk_nif.NifFile(path).shapes:
        sh = mk_nif.read_shape(s)
        for side, m in (("L", sh["v"][:, 0] < 0), ("R", sh["v"][:, 0] >= 0)):
            if not m.any():
                continue
            w = np.round(sh["w"][m], 4)
            rows = np.unique(w, axis=0)
            used = [sh["bones"][j].replace("NPC ", "") for j in np.nonzero(rows.max(0) > 0)[0]]
            print("%-18s %-14s %s: %5d verts, %d distinct weight sets, bones %s" % (
                os.path.basename(path), sh["name"], side, m.sum(), len(rows), used))
