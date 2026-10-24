"""N6: second blind round, literal vs rewrite only, on eight new prompts (see prediction_n6.txt).

Usage on the Mac mini, from ~/acestep-f0:
    ACE-Step-1.5/.venv/bin/python ~/acestep-mlx/bench/n6/n6.py clips <out dir>
    ACE-Step-1.5/.venv/bin/python ~/acestep-mlx/bench/n6/n6.py assemble <out dir> <test dir> <key.json>
``clips`` runs the N5 clip maker with these prompts and jobs (log: <out dir>/n6_0.6B.log);
``assemble`` builds the level-matched, shuffled test with the N5 vote page.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "n5"))

PROMPTS = {
    "pop": "canzone pop italiana estiva con voce maschile, 120 bpm",
    "ambient": "dark ambient drone, slow evolving pads",
    "dnb": "drum and bass, 174 bpm, reese bass, amen breaks",
    "blues": "acoustic blues with slide guitar and a raspy male voice",
    "lofi": "lo-fi hip hop beat, 85 bpm, dusty vinyl, mellow piano",
    "score": "orchestral film score, epic strings and brass, D minor",
    "reggae": "reggae, 75 bpm, female vocals, offbeat guitar skank",
    "indus": "techno industriale, 140 bpm, kick distorto, rumore metallico",
}
SHUFFLE_SEED = 20261024


def jobs(_lm):
    out = []
    for p in PROMPTS:
        out += [(f"K3-{p}-rewrite", p, "base"), (f"K3-{p}-literal", p, "literal")]
    return out + [("N1-lofi", "lofi", "base")]


def pairs():
    out = [(f"K3-{p}", p, (f"K3-{p}-rewrite", "rewrite"), (f"K3-{p}-literal", "literal")) for p in PROMPTS]
    return out + [("N1-lofi", "lofi", ("N1-lofi", "same file"), ("N1-lofi", "same file"))]


def main() -> None:
    command = sys.argv[1]
    if command == "clips":
        import make_clips

        make_clips.PROMPTS, make_clips.jobs = PROMPTS, jobs
        out = sys.argv[2]
        sys.argv = [sys.argv[0], "0.6B", os.path.join(out, "0.6B")]
        make_clips.main()
    elif command == "assemble":
        import assemble

        clips_dir, test_dir, key_path = sys.argv[2:5]
        clips = assemble.read_clips(clips_dir, lms=("0.6B",), log_prefix="n6")
        assemble.build(clips, test_dir, key_path, PROMPTS, pairs(), SHUFFLE_SEED, "N6")
    else:
        raise SystemExit(f"unknown command {command!r}")


if __name__ == "__main__":
    main()
