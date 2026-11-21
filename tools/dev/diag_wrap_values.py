"""The player's in-game slider values: the player's BodySlide base preset (SkinnyMorph 100 at weight 0, 0 at
weight 100) plus the OBody/RaceMenu body morphs read from the save (tools/dev/save_morphs.py)."""
import json


def player_values(path, weight):
    m = json.load(open(path))
    vals = {"SkinnyMorph": 1.0 - weight}
    for s, kv in m.items():
        if s.startswith("XPMSE") or s == "obody_processed":
            continue
        vals[s] = vals.get(s, 0.0) + sum(kv.values())
    return vals
