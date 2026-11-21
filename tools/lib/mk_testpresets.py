"""The body presets the clipping tests and the preview renders use (tools/clip_test.py, tools/qa_report.py and
tools/dev). The build never uses them: it fits the outfit to every UBE preset installed (mk_preset.corpus).

Personal presets are listed in tools/presets.local.json, which is not versioned (start from
tools/presets.example.json):
  preset_files  BodySlide preset files to read, relative to the Mod Organizer 2 mods folder, each with the names of
                the presets to take from it (null: all of them)
  test_presets  the presets of the clipping test, in the order of the report's columns
  pose_presets  the presets of the animation pose tests
  main_preset   the default preset of the preview renders
  tool_presets  the presets of single development tools ({"tool": [preset, ...]})
  labels        how docs/QA.md and the tools' output name each preset ({"preset name": "label"})
Without that file every test uses UBE's zeroed sliders."""
import json
import os

ZEROED = "-Zeroed Sliders-"
LOCAL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "presets.local.json")

_cfg = {}
if os.path.exists(LOCAL):
    with open(LOCAL, encoding="utf-8") as f:
        _cfg = json.load(f)

PRESET_FILES = [(entry["path"], entry.get("names")) for entry in _cfg.get("preset_files", [])]
TEST_PRESETS = list(_cfg.get("test_presets", [ZEROED]))
POSE_PRESETS = list(_cfg.get("pose_presets", [ZEROED]))
MAIN_PRESET = _cfg.get("main_preset", ZEROED)
LABELS = dict(_cfg.get("labels", {}))


def for_tool(tool, default=None):
    """The presets of one development tool: tool_presets[tool] from the local file, else `default` (else the main
    preset)."""
    return list(_cfg.get("tool_presets", {}).get(tool, [MAIN_PRESET] if default is None else default))


def label(text):
    """`text` with every preset name replaced by its label (also inside keys such as "<preset> @100")."""
    for name in sorted(LABELS, key=len, reverse=True):
        text = text.replace(name, LABELS[name])
    return text
