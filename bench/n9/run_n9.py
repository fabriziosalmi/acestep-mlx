"""N9: own lyrics through the app's engine, spoken to over the same JSON-lines protocol.

Usage: <app runtime python> run_n9.py <runtime dir> <project dir> <out dir> [literal]
Prints one "N9 ..." line per request and the song.json facts the prediction checks.
"""

import json
import os
import subprocess
import sys
import time

LYRICS_IT = """[Verse]
Sul porto di Genova scende la sera
le barche si accendono una per una
il vento ci porta l'odore del mare
e io resto qui ad aspettare la luna
[Chorus]
Canta con me, canta piano
la notte e' lunga, il giorno e' lontano
canta con me sotto le stelle
finche' la luce non tocca la pelle"""

LYRICS_EN = """[Verse]
Concrete walls are shaking tonight
red lights flashing, hold on tight
[Chorus]
We don't stop, we don't sleep
the bass goes on, the floor runs deep"""

REQUESTS = [
    ("baseline", "una ballata folk italiana con chitarra acustica e voce femminile", {}),
    ("it", "una ballata folk italiana con chitarra acustica e voce femminile", {"lyrics": LYRICS_IT}),
    ("en", "dark techno, 140 bpm", {"lyrics": LYRICS_EN}),
    ("blank", "dark techno", {"lyrics": "  \n "}),
    ("vocals-off", "dark techno", {"lyrics": "la la", "vocals": False}),
]

# Second round, after round 1 showed the LM's caption drifting from the prompt with own lyrics
# ("dark techno" became "pop-punk"): the same lyrics with the prompt as the caption.
LITERAL = [
    ("it-literal", REQUESTS[1][1], {"lyrics": LYRICS_IT, "literal": True}),
    ("en-literal", REQUESTS[2][1], {"lyrics": LYRICS_EN, "literal": True}),
]


def main() -> None:
    """Start the engine, send every request in order, report each outcome."""
    runtime, project, out = sys.argv[1:4]
    requests = LITERAL if sys.argv[4:] == ["literal"] else REQUESTS
    env = {**os.environ, "PYTHONPATH": f"{runtime}/engine:{runtime}/upstream", "PYTHONNOUSERSITE": "1",
           "PYTHONUNBUFFERED": "1", "PYTHONPYCACHEPREFIX": os.path.expanduser("~/Library/Caches/ACE-Step/pycache")}
    env.pop("PYTHONHOME", None)
    log = open(os.path.join(out, "engine_stderr.log"), "w")
    engine = subprocess.Popen(
        [f"{runtime}/python/bin/python3", "-m", "acestep_mlx.engine_service", "--project", project,
         "--out", out, "--preset", "mini"],
        cwd=project, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True)

    def event() -> dict:
        return json.loads(engine.stdout.readline())

    started = time.time()
    while (first := event())["event"] != "ready":
        if first["event"] == "error":
            sys.exit(f"N9 engine failed: {first}")
    print(f"N9 ready in {time.time() - started:.1f} s", flush=True)
    for name, prompt, options in requests:
        asked = time.time()
        engine.stdin.write(json.dumps({"id": f"n9-{name}", "cmd": "generate", "prompt": prompt,
                                       "options": options}) + "\n")
        engine.stdin.flush()
        while (ev := event())["event"] == "phase":
            pass
        took = time.time() - asked
        if ev["event"] == "done":
            song = ev["song"]
            given = options.get("lyrics")
            print("N9 " + json.dumps({
                "request": name, "took_s": round(took, 1), "instrumental": song["instrumental"],
                "bpm": song["bpm"], "keyscale": song["keyscale"], "language": song["language"],
                "lyrics_exact": None if given is None else song["lyrics"] == given.strip(),
                "sources": song["sources"], "caption": song["caption"][:120], "audio": song["audio"],
                "dir": os.path.basename(song["dir"]),
            }, ensure_ascii=False), flush=True)
        else:
            print("N9 " + json.dumps({"request": name, "took_s": round(took, 1), "error": ev.get("message")}),
                  flush=True)
    engine.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
    engine.stdin.flush()
    engine.wait(timeout=60)


if __name__ == "__main__":
    main()
