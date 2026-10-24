"""Turn the N5 clips into the blind test folder plus the key that unblinds it.

Usage on the Mac mini, after run_n5.sh:
    ACE-Step-1.5/.venv/bin/python ~/acestep-mlx/bench/n5/assemble.py <clips dir> <test dir> <key.json>
The test dir gets index.html and clips/NNa.wav, NNb.wav (16-bit PCM); key.json maps each pair to its
clips and stays out of the test dir. Pair order and the A/B side come from a fixed seed. Within a pair
both clips are scaled down to the same RMS (no clip is boosted, peaks stay at or below 0.99), so
loudness does not decide the vote.
"""

import json
import os
import random
import sys

import numpy as np
import soundfile as sf

SHUFFLE_SEED = 20261023
PEAK_MAX = 0.99
HERE = os.path.dirname(os.path.abspath(__file__))


def read_clips(clips_dir: str, lms=("0.6B", "1.7B"), log_prefix: str = "n5") -> dict:
    """Clip id -> N5 record, from the series logs."""
    clips = {}
    for lm in lms:
        with open(os.path.join(clips_dir, f"{log_prefix}_{lm}.log")) as fh:
            for line in fh:
                if line.startswith("N5 {"):
                    record = json.loads(line[3:])
                    record["path"] = os.path.join(clips_dir, lm, record["audio"])
                    clips[record["clip"]] = record
    return clips


def pairs(prompts) -> list:
    """(pair id, prompt key, (clip, side label), (clip, side label))."""
    out = []
    for p in prompts:
        out += [(f"K1-{p}", p, (f"K1-{p}-s1", "shift 1.0"), (f"K1-{p}-s3", "shift 3.0")),
                (f"K2-{p}", p, (f"K2-{p}-06", "LM 0.6B"), (f"K2-{p}-17", "LM 1.7B")),
                (f"K3-{p}", p, (f"K3-{p}-rewrite", "rewrite"), (f"K3-{p}-literal", "literal"))]
    for p in ("rock", "house", "tekno"):
        out.append((f"K4-{p}", p, (f"K4-{p}-forced", "forced"), (f"K4-{p}-free", "free")))
    out.append(("N1-house", "house", ("N1-house", "same file"), ("N1-house", "same file")))
    out.append(("N2-rock", "rock", ("N2-rock-a", "run a"), ("N2-rock-b", "run b")))
    return out


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def build(clips: dict, test_dir: str, key_path: str, prompts: dict, order: list, shuffle_seed: int,
          tag: str) -> None:
    """Write the shuffled, level-matched pairs, the vote page tagged ``tag`` and the key."""
    rng = random.Random(shuffle_seed)
    order = list(order)
    rng.shuffle(order)
    os.makedirs(os.path.join(test_dir, "clips"), exist_ok=True)
    key, page = {}, []
    for index, (pair_id, prompt_key, first, second) in enumerate(order, start=1):
        number = f"{index:02d}"
        sides = [first, second] if rng.random() < 0.5 else [second, first]
        audio = [sf.read(clips[clip]["path"], dtype="float32", always_2d=True) for clip, _ in sides]
        levels = [rms(a) for a, _ in audio]
        peaks = [float(np.max(np.abs(a))) for a, _ in audio]
        target = min(lv * min(1.0, PEAK_MAX / pk) for lv, pk in zip(levels, peaks))
        entry = {"pair": pair_id, "prompt": prompts[prompt_key]}
        for letter, (clip, label), (data, rate), level in zip("ab", sides, audio, levels):
            gain = target / level
            sf.write(os.path.join(test_dir, "clips", f"{number}{letter}.wav"), data * gain, rate,
                     subtype="PCM_16")
            entry[letter.upper()] = {"clip": clip, "side": label, "gain_db": round(20 * np.log10(gain), 2),
                                     "md5": clips[clip]["md5"], "bpm": clips[clip]["bpm"],
                                     "keyscale": clips[clip]["keyscale"]}
        key[number] = entry
        page.append({"n": number, "prompt": prompts[prompt_key]})
    with open(key_path, "w") as fh:
        json.dump({"shuffle_seed": shuffle_seed, "pairs": key}, fh, ensure_ascii=False, indent=1)
    with open(os.path.join(HERE, "vote_page.html")) as fh:
        html = (fh.read().replace("/*PAIRS*/[]", json.dumps(page, ensure_ascii=False))
                .replace('/*TAG*/"N5"', json.dumps(tag)))
    with open(os.path.join(test_dir, "index.html"), "w") as fh:
        fh.write(html)
    print(f"{len(order)} pairs written to {test_dir}; key in {key_path}")


def main() -> None:
    clips_dir, test_dir, key_path = sys.argv[1:4]
    sys.path.insert(0, HERE)
    from make_clips import PROMPTS

    build(read_clips(clips_dir), test_dir, key_path, PROMPTS, pairs(PROMPTS), SHUFFLE_SEED, "N5")


if __name__ == "__main__":
    main()
