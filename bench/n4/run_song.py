"""One song through LeanEngine with a chosen LM and duration, timed per phase.

Usage on the Mac mini, from ~/acestep-f0, under run_n4.sh (time -l + footprint sampler):
    ACE-Step-1.5/.venv/bin/python ~/acestep-mlx/bench/n4/run_song.py <lm_model> <seconds> <tag> [prompt]
Prints one "N4 {json}" line: load time, phase start times, MLX peak per phase, the song's
duration, bpm and key as written, and the audio file's measured length.
"""

import json
import os
import sys
import time

HOME = os.path.expanduser("~")
PROJECT = f"{HOME}/acestep-f0/ACE-Step-1.5"
sys.path[:0] = [f"{HOME}/acestep-mlx/engine", PROJECT]

DEFAULT_PROMPT = "dark techno, 140 bpm in A minor, rolling bassline"


def main() -> None:
    """Load, make one song, print the measurements."""
    lm_model, seconds, tag = sys.argv[1], float(sys.argv[2]), sys.argv[3]
    prompt = sys.argv[4] if len(sys.argv) > 4 else DEFAULT_PROMPT
    import mlx.core as mx
    import soundfile as sf

    from acestep_mlx.engine_service import LeanEngine
    from acestep_mlx.recipe import Preset

    t0 = time.time()
    preset = Preset("bench", lm_model, default_duration=seconds, max_duration=600.0)
    engine = LeanEngine(PROJECT, f"{HOME}/acestep-f0/n4_songs/{tag}", preset)
    load_s = time.time() - t0
    phases, peaks = {}, {}

    def report(phase):
        if phase not in phases:
            peaks[f"before_{phase}"] = round(mx.get_peak_memory() / 2**20)
            mx.reset_peak_memory()
            phases[phase] = round(time.time() - t_song, 2)

    t_song = time.time()
    song = engine.generate(f"{tag}-0000", prompt, {}, report)
    song_s = time.time() - t_song
    peaks["after_rendering"] = round(mx.get_peak_memory() / 2**20)
    info = sf.info(os.path.join(song["dir"], song["audio"]))
    print("N4 " + json.dumps({
        "tag": tag, "lm_model": lm_model, "requested_s": seconds, "load_s": round(load_s, 1),
        "song_s": round(song_s, 1), "phase_start_s": phases, "mlx_peak_mb": peaks,
        "audio_s": round(info.duration, 3), "subtype": info.subtype, "bpm": song["bpm"],
        "keyscale": song["keyscale"], "instrumental": song["instrumental"],
        "lyrics_lines": len((song["lyrics"] or "").splitlines()),
    }), flush=True)


if __name__ == "__main__":
    main()
