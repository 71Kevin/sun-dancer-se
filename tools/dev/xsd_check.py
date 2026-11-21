"""Validates SMP system files against FSMP's schema (plain Python, needs `pip install xmlschema`):
  python tools/dev/xsd_check.py "<mods>\\Faster HDT-SMP\\SKSE\\Plugins\\hdtSkinnedMeshConfigs\\hdtSMP64.xsd" <xml> [<xml> ...]
The schema's target namespace is "FSMP-Validator" while the game's files carry none, so it is applied in memory."""
import re
import sys

import xmlschema

xsd = xmlschema.XMLSchema(sys.argv[1])
for path in sys.argv[2:]:
    text = open(path, encoding="utf-8-sig").read()
    text = re.sub(r"<system\b", '<system xmlns="FSMP-Validator"', text, count=1)
    errs = list(xsd.iter_errors(text))
    print("%s: %s" % (path.replace("/", "\\").split("\\")[-1], "VALID" if not errs else "%d errors" % len(errs)))
    for e in errs[:8]:
        print("   ", str(e.reason)[:220], "| path:", e.path)
