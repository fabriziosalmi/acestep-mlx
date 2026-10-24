"""N8 P4: one song with every random generator seeded, to compare two installs sample by sample.

Usage: <python> same_seed.py <project with checkpoints/> <out dir>
Run once with the development venv and once with the app's runtime (PYTHONPATH set to the
engine and upstream folders); equal ``samples_md5`` means the same audio.
"""

import hashlib
import json
import random
import sys

import mlx.core as mx
import numpy as np
import soundfile as sf
import torch

from acestep_mlx import engine_service
from acestep_mlx.recipe import PRESETS

PROMPT = "a short instrumental lo-fi beat with soft piano, 80 bpm"


def main() -> None:
    """Seed everything, make one song, print its identity as JSON."""
    project, out = sys.argv[1], sys.argv[2]
    random.seed(8)
    np.random.seed(8)
    torch.manual_seed(8)
    mx.random.seed(8)
    engine = engine_service.LeanEngine(project, out, PRESETS["mini"])
    song = engine.generate("n8-same-seed", PROMPT, {}, lambda phase: None)
    samples, rate = sf.read(f"{song['dir']}/{song['audio']}", dtype="float32")
    print("N8_RESULT " + json.dumps({
        "python": sys.executable, "seed": song["seed"], "bpm": song["bpm"], "rate": rate,
        "frames": len(samples), "caption_md5": hashlib.md5(song["caption"].encode()).hexdigest(),
        "lyrics_md5": hashlib.md5(song["lyrics"].encode()).hexdigest(),
        "samples_md5": hashlib.md5(samples.tobytes()).hexdigest(),
    }))


if __name__ == "__main__":
    main()
