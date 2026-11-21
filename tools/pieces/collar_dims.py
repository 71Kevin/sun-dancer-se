"""The Collar torque's outline, shared by piece_collar (meshes, in Blender) and tools/make_textures.py (the fluting
and the piercings follow its edges; plain Python, no Blender modules here).

Geodesic distance from the neck opening of the body mesh (inner edge, outer edge) by angle round the neck (0 =
front): about 3.1 wide under the throat (where it lies on the slope from the neck to the chest, seen from the front
about two thirds of that), 1.8 at the sides, 0.9 at the nape."""
import numpy as np

INNER = [(0, 2.6), (1.0, 2.6), (1.57, 2.7), (2.4, 2.75), (np.pi, 2.7)]
OUTER = [(0, 5.7), (0.5, 5.5), (1.0, 5.0), (1.57, 4.5), (2.2, 4.0), (np.pi, 3.6)]
