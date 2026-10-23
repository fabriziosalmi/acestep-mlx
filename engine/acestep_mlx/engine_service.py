"""Engine process for the macOS app: loads the lean DiT handler and the LM once, then serves.

Run with the upstream venv's Python:
    python -m acestep_mlx.engine_service --project <ACE-Step-1.5 dir> --out <songs dir>
Each song goes to ``<out>/<timestamp>-<id>/`` as the FLAC plus ``song.json``. The
protocol uses the original stdout; fd 1 is pointed at stderr before anything is
imported, so upstream prints, tqdm and C-level output never corrupt it.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

from acestep_mlx.protocol import Emitter, serve

# Measured envelope on a 16 GB Mac (bench/n3): 30 s songs peak at 11.8 GB. Longer
# songs are not measured yet.
SONG_SECONDS = 30.0
DIT_MODEL = "acestep-v15-turbo"
LM_MODEL = "acestep-5Hz-lm-0.6B"
# generate_music reports LM work below this fraction and DiT/VAE work from it on.
RENDER_FROM = 0.51


def song_dir_name(request_id: str, now: datetime) -> str:
    """``YYYYmmdd-HHMMSS-<first 8 chars of the id>``, sortable by creation time."""
    return f"{now.strftime('%Y%m%d-%H%M%S')}-{request_id[:8]}"


class LeanEngine:
    """Describe-to-song pipeline with models kept loaded between requests."""

    def __init__(self, project: str, out_dir: str):
        """Load the DiT handler and the 5 Hz LM from ``<project>/checkpoints``.

        Raises:
            RuntimeError: If either model fails to initialize.
        """
        from acestep.llm_inference import LLMHandler

        from acestep_mlx.lean_handler import LeanAceStepHandler

        self.out_dir = out_dir
        self.dit = LeanAceStepHandler()
        status, ok = self.dit.initialize_service(project, DIT_MODEL, device="auto")
        if not ok:
            raise RuntimeError(f"DiT init failed: {str(status)[-300:]}")
        self.llm = LLMHandler()
        status, ok = self.llm.initialize(
            os.path.join(project, "checkpoints"), LM_MODEL, backend="mlx", device="auto"
        )
        if not ok:
            raise RuntimeError(f"LM init failed: {str(status)[-300:]}")

    def generate(self, request_id: str, prompt: str, report_phase) -> dict:
        """Write lyrics and metadata with the LM, then render; returns the song record.

        Raises:
            RuntimeError: If sample creation or generation reports failure.
        """
        from acestep.inference import (GenerationConfig, GenerationParams, create_sample,
                                       generate_music)

        started = time.time()
        report_phase("lyrics")
        sample = create_sample(self.llm, prompt)
        if not sample.success:
            raise RuntimeError(f"lyrics and metadata failed: {sample.error}")
        report_phase("composing")

        def progress(value, *_args, **_kwargs):
            report_phase("rendering" if value >= RENDER_FROM else "composing")

        now = datetime.now(timezone.utc)
        song_dir = os.path.join(self.out_dir, song_dir_name(request_id, now.astimezone()))
        params = GenerationParams(
            caption=sample.caption, lyrics=sample.lyrics, bpm=sample.bpm,
            duration=SONG_SECONDS, keyscale=sample.keyscale,
            vocal_language=sample.language or "unknown", thinking=True,
        )
        config = GenerationConfig(batch_size=1, use_random_seed=True, audio_format="flac")
        result = generate_music(self.dit, self.llm, params, config, save_dir=song_dir,
                                progress=progress)
        if not result.success or not result.audios:
            raise RuntimeError(f"generation failed: {result.error}")
        audio = result.audios[0]
        song = {
            "prompt": prompt, "caption": sample.caption, "lyrics": sample.lyrics,
            "bpm": sample.bpm, "keyscale": sample.keyscale, "language": sample.language,
            "duration": SONG_SECONDS, "seed": audio.get("params", {}).get("seed"),
            "audio": os.path.basename(audio["path"]), "created": now.isoformat(),
            "render_s": round(time.time() - started, 1),
        }
        with open(os.path.join(song_dir, "song.json"), "w") as fh:
            json.dump(song, fh, ensure_ascii=False, indent=1)
        return {**song, "dir": song_dir}


def protocol_stdout():
    """Keep the real stdout for the protocol and send everything else on fd 1 to stderr."""
    channel = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return channel


def main(argv=None) -> int:
    """Parse arguments, load models and serve requests from stdin."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project", required=True, help="ACE-Step-1.5 dir with checkpoints/")
    parser.add_argument("--out", required=True, help="directory that receives the songs")
    args = parser.parse_args(argv)
    emitter = Emitter(protocol_stdout())
    sys.path.insert(0, args.project)
    os.makedirs(args.out, exist_ok=True)
    started = time.time()
    try:
        engine = LeanEngine(args.project, args.out)
    except Exception as exc:  # noqa: BLE001 - the app shows it
        emitter.emit("error", id=None, message=f"engine failed to load: {exc}")
        return 1
    serve(engine, sys.stdin, emitter, load_s=round(time.time() - started, 1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
