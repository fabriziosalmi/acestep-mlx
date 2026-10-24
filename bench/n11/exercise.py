"""N11: drive every engine path in one process, then say which distributions were needed.

Usage: <runtime python> exercise.py <project with checkpoints/> <out dir>
Paths: the Hugging Face listing, download and sha256 check of one small LFS file; engine load;
songs as wav and mp3, instrumental and with vocals, own lyrics, literal, 60 s, and another like
the first. Then every installed distribution is classed: imported (a module of it is in
sys.modules), required (named, without an extra marker, by an imported or required one), or
removable, with its size on disk. Writes <out dir>/n11.json and prints a summary.
"""

import importlib.metadata as md
import json
import os
import re
import sys
import time

from acestep_mlx import engine_service, models
from acestep_mlx.recipe import PRESETS

LYRICS = "[Verse]\nSul porto di Genova scende la sera\nle barche si accendono una per una"
REQUESTS = [
    ("wav-instrumental", "lo-fi hip hop beat, 85 bpm, mellow piano", {}),
    ("mp3-vocals", "a warm Italian folk ballad with a female voice", {"format": "mp3"}),
    ("own-lyrics", "una ballata folk italiana", {"lyrics": LYRICS}),
    ("literal-60s", "dark techno, 140 bpm, no vocals", {"literal": True, "duration": 60}),
]


def name_of(requirement: str) -> str:
    """The normalized distribution name a requirement string starts with."""
    return re.split(r"[\s;<>=!~\[(]", requirement.strip(), maxsplit=1)[0].lower().replace("_", "-")


def norm(dist: str) -> str:
    return dist.lower().replace("_", "-")


def main() -> None:
    project, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    timings = {}

    probe = models.Source("probe", models.MAIN.repo, models.MAIN.revision,
                          ("acestep-v15-turbo/silence_latent.pt",))
    files = models.remote_files(probe)
    target = os.path.join(out, "probe")
    models.fetch(probe, target)
    models.check(target, files)
    timings["hf_probe_files"] = [(f.path, f.size, bool(f.sha256)) for f in files]

    started = time.time()
    engine = engine_service.LeanEngine(project, out, PRESETS["mini"])
    timings["load_s"] = round(time.time() - started, 1)
    first = None
    for name, prompt, options in REQUESTS:
        t = time.time()
        song = engine.generate(f"n11-{name}", prompt, options, lambda phase: None)
        timings[name] = round(time.time() - t, 1)
        first = first or song
    t = time.time()
    engine.generate("n11-like", first["prompt"], {"like": os.path.basename(first["dir"])}, lambda phase: None)
    timings["like"] = round(time.time() - t, 1)

    top_level = md.packages_distributions()
    imported = set()
    for module in list(sys.modules):
        for dist in top_level.get(module.split(".")[0], []):
            imported.add(norm(dist))
    installed = {norm(d.metadata["Name"]): d for d in md.distributions()}
    keep = set(imported)
    frontier = list(imported)
    while frontier:
        dist = installed.get(frontier.pop())
        for req in (dist.requires or []) if dist else []:
            if "extra ==" in req.replace("extra==", "extra =="):
                continue
            name = name_of(req)
            if name in installed and name not in keep:
                keep.add(name)
                frontier.append(name)

    def size(dist) -> int:
        return sum(os.path.getsize(f.locate()) for f in (dist.files or []) if os.path.isfile(f.locate()))

    report = {
        "timings": timings,
        "imported": sorted(imported & set(installed)),
        "required_only": sorted(keep - imported),
        "removable": sorted(((n, size(installed[n])) for n in set(installed) - keep), key=lambda x: -x[1]),
    }
    with open(os.path.join(out, "n11.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    removable_mb = sum(s for _n, s in report["removable"]) / 1e6
    print(f"N11 installed {len(installed)} imported {len(report['imported'])} "
          f"required-only {len(report['required_only'])} removable {len(report['removable'])} "
          f"({removable_mb:.0f} MB)")
    print("N11 timings " + json.dumps(timings))


if __name__ == "__main__":
    main()
