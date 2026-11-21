"""Reads (read-only) the player's RaceMenu/OBody body morphs from an SKSE co-save and writes them as extra body
samples for the fitting and the tests (<work>/extra_samples.json, not versioned):

  python tools/dev/save_morphs.py <save.skse> <work> [--base "SkinnyMorph=1:0"]

The in-game body is the player's BodySlide base preset (SkinnyMorph 100 at weight 0, 0 at weight 100; see --base)
plus the morphs, so two samples are written (character weight 0 and 100).

Co-save layout used: header 'SKSE' + 4 u32 (format, skse, runtime, plugin count); per plugin u32 uid, chunk count,
length; per chunk u32 type, version, length. Plugin 'SKEE' (RaceMenu): STTB string table (u32 count, u16-prefixed
strings), then per actor a 'MRPH' chunk (u32 form id) followed by 'MRDT' (u32 morph count; per morph u32 name index,
u32 key count; per key u32 key index, f32 value)."""
import json
import os
import struct
import sys


def chunks(data, plugin=b"EEKS"):
    n = struct.unpack_from("<I", data, 16)[0]
    i = 20
    for _ in range(n):
        uid, count, length = struct.unpack_from("<3I", data, i)
        i += 12
        if struct.pack("<I", uid) == plugin:
            j = i
            for _ in range(count):
                ctype, ver, clen = struct.unpack_from("<3I", data, j)
                yield struct.pack("<I", ctype)[::-1].decode("latin-1"), ver, data[j + 12:j + 12 + clen]
                j += 12 + clen
        i += length


def player_morphs(path, form_id=0x14):
    data = open(path, "rb").read()
    if data[:4] != b"SKSE":
        raise ValueError("not an SKSE co-save: %s" % path)
    strings, morphs, take = [], None, False
    for ctype, ver, payload in chunks(data):
        if ctype == "STTB":
            count = struct.unpack_from("<I", payload, 0)[0]
            p = 4
            for _ in range(count):
                n = struct.unpack_from("<H", payload, p)[0]
                strings.append(payload[p + 2:p + 2 + n].decode("latin-1"))
                p += 2 + n
        elif ctype == "MRPH" and len(payload) == 4:
            take = struct.unpack_from("<I", payload, 0)[0] == form_id
        elif ctype == "MRDT" and take:
            q = 4
            morphs = {}
            for _ in range(struct.unpack_from("<I", payload, 0)[0]):
                ni, nk = struct.unpack_from("<II", payload, q)
                q += 8
                keys = {}
                for _ in range(nk):
                    ki, val = struct.unpack_from("<If", payload, q)
                    q += 8
                    keys[strings[ki]] = val
                morphs[strings[ni]] = keys
            take = False
    if morphs is None:
        raise ValueError("no body morphs for form %X in %s" % (form_id, path))
    return morphs


def main():
    save, work = sys.argv[1], sys.argv[2]
    morphs = player_morphs(save)
    vals = {}
    for s, kv in morphs.items():
        if s.startswith("XPMSE") or s == "obody_processed":
            continue
        total = sum(kv.values())
        if total:
            vals[s] = total
    out = [{"name": "In-game character (weight 0)", "values": dict(vals, SkinnyMorph=1.0)},
           {"name": "In-game character (weight 100)", "values": dict(vals, SkinnyMorph=0.0)}]
    path = os.path.join(work, "extra_samples.json")
    json.dump(out, open(path, "w", encoding="utf-8"), indent=1)
    print("%d morphed sliders -> %s" % (len(vals), path))


if __name__ == "__main__":
    main()
