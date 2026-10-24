"""Engine process for the macOS app: loads the lean DiT handler and the LM once, then serves.

Run with the upstream venv's Python:
    python -m acestep_mlx.engine_service --project <dir> --out <songs dir> [--preset mini]
Each song goes to ``<out>/<timestamp>-<id>/`` as the audio file plus ``song.json``; the
rules that shape it are in ``recipe.py``. The
protocol uses the original stdout; fd 1 is pointed at stderr before anything is
imported, so upstream prints, tqdm and C-level output never corrupt it.
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone

from acestep_mlx import models
from acestep_mlx.protocol import Emitter, serve
from acestep_mlx.recipe import (PRESETS, UPSTREAM_FORMAT, Options, Plan, Preset, parse_options,
                                plan_request, song_folder)

DIT_MODEL = "acestep-v15-turbo"
# generate_music reports LM work below this fraction and DiT/VAE work from it on.
RENDER_FROM = 0.51
INSTRUMENTAL_LYRICS = "[Instrumental]"
SEED_MAX = 2**32 - 1


def song_dir_name(request_id: str, now: datetime) -> str:
    """``YYYYmmdd-HHMMSS-<first 8 chars of the id>``, sortable by creation time."""
    return f"{now.strftime('%Y%m%d-%H%M%S')}-{request_id[:8]}"


class LeanEngine:
    """Describe-to-song pipeline with models kept loaded between requests."""

    def __init__(self, project: str, out_dir: str, preset: Preset):
        """Load the DiT handler and the preset's 5 Hz LM from ``<project>/checkpoints``.

        Raises:
            RuntimeError: If either model fails to initialize.
        """
        from acestep.llm_inference import LLMHandler

        from acestep_mlx.lean_handler import LeanAceStepHandler

        self.out_dir, self.preset = out_dir, preset
        self.dit = LeanAceStepHandler()
        status, ok = self.dit.initialize_service(project, DIT_MODEL, device="auto")
        if not ok:
            raise RuntimeError(f"DiT init failed: {str(status)[-300:]}")
        self.llm = LLMHandler()
        status, ok = self.llm.initialize(
            os.path.join(project, "checkpoints"), preset.lm_model, backend="mlx", device="auto"
        )
        if not ok:
            raise RuntimeError(f"LM init failed: {str(status)[-300:]}")

    def _text_from_lm(self, prompt: str, plan: Plan, report_phase) -> dict:
        """Caption, lyrics and metadata from the LM, with the plan's overrides.

        With the user's own lyrics the LM only describes them (``format_sample``: caption,
        BPM, key, language) and the lyrics are sung exactly as written; otherwise it writes
        them too (``create_sample``).
        """
        from acestep.inference import create_sample, format_sample

        report_phase("lyrics")
        if plan.lyrics is not None:
            forced = {"bpm": plan.bpm, "keyscale": plan.keyscale, "language": plan.language}
            sample = format_sample(self.llm, prompt, plan.lyrics,
                                   user_metadata={k: v for k, v in forced.items() if v} or None)
            if not sample.success:
                raise RuntimeError(f"metadata for the lyrics failed: {sample.error}")
            return {
                "caption": prompt if plan.literal else sample.caption,
                "lyrics": plan.lyrics,
                "instrumental": False,
                "bpm": plan.bpm or sample.bpm,
                "keyscale": plan.keyscale or sample.keyscale,
                # The LM named the lyrics' language right in 2 of 4 songs (bench/n9).
                "language": plan.language or sample.language,
                "duration": plan.duration,
            }
        sample = create_sample(self.llm, prompt, instrumental=plan.instrumental)
        if not sample.success:
            raise RuntimeError(f"lyrics and metadata failed: {sample.error}")
        return {
            "caption": prompt if plan.literal else sample.caption,
            # The LM's metadata can override the instrumental flag; the plan decides.
            "lyrics": INSTRUMENTAL_LYRICS if plan.instrumental else sample.lyrics,
            "instrumental": plan.instrumental,
            "bpm": plan.bpm or sample.bpm,
            "keyscale": plan.keyscale or sample.keyscale,
            "language": sample.language,
            "duration": plan.duration,
        }

    def _text_from_song(self, name: str, opts: Options) -> dict:
        """Caption, lyrics and metadata of an existing song, for another version of it."""
        with open(os.path.join(song_folder(self.out_dir, name), "song.json")) as fh:
            base = json.load(fh)
        text = {k: base.get(k) for k in ("caption", "lyrics", "bpm", "keyscale", "language")}
        text["instrumental"] = base.get(
            "instrumental", (base.get("lyrics") or "").strip() == INSTRUMENTAL_LYRICS
        )
        text["duration"] = float(
            opts.duration or base.get("duration") or self.preset.default_duration
        )
        return text

    def generate(self, request_id: str, prompt: str, options: dict, report_phase) -> dict:
        """Make one song following the recipe; returns the song record.

        Raises:
            ValueError: For invalid options.
            RuntimeError: If sample creation or generation reports failure.
        """
        from acestep.inference import GenerationConfig, GenerationParams, generate_music

        from acestep_mlx.audio_export import export_mp3, require_audio

        opts = parse_options(options, self.preset)
        plan = plan_request(prompt, opts, self.preset)
        started = time.time()
        if plan.like:
            text = self._text_from_song(plan.like, opts)
            sources = {key: "like" for key in plan.sources}
        else:
            text = self._text_from_lm(prompt, plan, report_phase)
            sources = plan.sources
        report_phase("composing")

        def progress(value, *_args, **_kwargs):
            report_phase("rendering" if value >= RENDER_FROM else "composing")

        now = datetime.now(timezone.utc)
        name = song_dir_name(request_id, now.astimezone())
        song_dir = os.path.join(self.out_dir, name)
        params = GenerationParams(
            caption=text["caption"], lyrics=text["lyrics"], instrumental=text["instrumental"],
            bpm=text["bpm"], keyscale=text["keyscale"] or "", duration=text["duration"],
            vocal_language=text["language"] or "unknown", thinking=True,
            use_cot_caption=not (plan.literal or plan.like), enable_normalization=False,
        )
        # The seed is drawn here and passed fixed: with use_random_seed=True upstream records
        # the LM's seed but lets the DiT draw another one, so song.json would name a seed
        # that does not reproduce the song.
        seed = random.randint(0, SEED_MAX)
        config = GenerationConfig(batch_size=1, use_random_seed=False, seeds=[seed],
                                  audio_format=UPSTREAM_FORMAT)
        result = generate_music(self.dit, self.llm, params, config, save_dir=song_dir,
                                progress=progress)
        if not result.success or not result.audios:
            raise RuntimeError(f"generation failed: {result.error}")
        audio = result.audios[0]
        path = require_audio(audio.get("path"))
        if plan.format == "mp3":
            path = export_mp3(path)
        song = {
            "prompt": prompt, **text, "literal": plan.literal, "sources": sources,
            "parent": plan.like, "format": opts.format, "preset": self.preset.name,
            "lm_model": self.preset.lm_model, "dit_model": DIT_MODEL,
            "seed": seed, "audio": os.path.basename(path),
            "created": now.isoformat(), "render_s": round(time.time() - started, 1),
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
    parser.add_argument("--preset", choices=sorted(PRESETS), default="mini")
    args = parser.parse_args(argv)
    emitter = Emitter(protocol_stdout())
    os.makedirs(args.out, exist_ok=True)
    preset = PRESETS[args.preset]
    try:
        models.ensure(os.path.join(args.project, "checkpoints"), models.sources_for(preset.lm_model),
                      lambda done, total, complete: emitter.emit(
                          "download", done=done, total=total, complete=complete))
    except Exception as exc:  # noqa: BLE001 - the app shows it
        emitter.emit("error", id=None, message=f"models could not be downloaded: {exc}")
        return 1
    started = time.time()
    try:
        engine = LeanEngine(args.project, args.out, preset)
    except Exception as exc:  # noqa: BLE001 - the app shows it
        emitter.emit("error", id=None, message=f"engine failed to load: {exc}")
        return 1
    serve(engine, sys.stdin, emitter, load_s=round(time.time() - started, 1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
