"""Clips for the N5 blind listening test, one LM per process.

Usage on the Mac mini, from ~/acestep-f0:
    ACE-Step-1.5/.venv/bin/python ~/acestep-mlx/bench/n5/make_clips.py <0.6B|1.7B> <out_dir> [first N jobs only]
Each clip goes to ``<out_dir>/<clip id>/`` (audio + song.json) and one "N5 {json}" line is printed per
clip. K1 clips are re-renders of a baseline song from its own LM codes, caption and metadata with a
fixed DiT seed, so only ``shift`` differs between the two sides.
"""

import dataclasses
import hashlib
import json
import os
import sys
import time

HOME = os.path.expanduser("~")
PROJECT = f"{HOME}/acestep-f0/ACE-Step-1.5"
sys.path[:0] = [f"{HOME}/acestep-mlx/engine", PROJECT]

PROMPTS = {
    "folk": "ballata folk italiana, chitarra acustica, voce femminile",
    "rock": "energetic pop rock song with male vocals, 128 bpm, D major",
    "house": "minimal house, 124 bpm, deep rolling groove",
    "tekno": "freetekno, 180 bpm, acid 303 line, distorted kick",
}
FORCED = ("rock", "house", "tekno")
SECONDS = 30.0


def jobs(lm: str):
    """(clip id, prompt key, mode) for this LM; mode is base, literal, free or k1."""
    if lm == "1.7B":
        return [(f"K2-{p}-17", p, "base") for p in PROMPTS]
    out = []
    for p in PROMPTS:
        out += [(f"K1-{p}", p, "k1"), (f"K2-{p}-06", p, "base"),
                (f"K3-{p}-rewrite", p, "base"), (f"K3-{p}-literal", p, "literal")]
    for p in FORCED:
        out += [(f"K4-{p}-forced", p, "base"), (f"K4-{p}-free", p, "free")]
    return out + [("N1-house", "house", "base"), ("N2-rock-a", "rock", "base"),
                  ("N2-rock-b", "rock", "base")]


def md5(path: str) -> str:
    """md5 of the decoded samples (the WAV header differs between identical renders)."""
    import soundfile as sf

    return hashlib.md5(sf.read(path, dtype="float32")[0].tobytes()).hexdigest()


def main() -> None:
    lm, out_dir = sys.argv[1], sys.argv[2]
    import acestep.inference as inference
    import soundfile as sf

    from acestep_mlx import engine_service
    from acestep_mlx.engine_service import LeanEngine
    from acestep_mlx.recipe import Preset

    captured = {}
    upstream_generate = inference.generate_music

    def capturing_generate(*args, **kwargs):
        result = upstream_generate(*args, **kwargs)
        captured["result"], captured["params"] = result, args[2]
        return result

    inference.generate_music = capturing_generate
    upstream_plan = engine_service.plan_request
    free = {"on": False}

    def plan_request(prompt, opts, preset):
        plan = upstream_plan(prompt, opts, preset)
        if not free["on"]:
            return plan
        sources = {k: v for k, v in plan.sources.items() if k not in ("bpm", "keyscale")}
        return dataclasses.replace(plan, bpm=None, keyscale=None, sources=sources)

    engine_service.plan_request = plan_request
    t0 = time.time()
    preset = Preset("bench", f"acestep-5Hz-lm-{lm}", default_duration=SECONDS, max_duration=SECONDS)
    engine = LeanEngine(PROJECT, out_dir, preset)
    print(f"N5 load {lm} {time.time() - t0:.1f} s", flush=True)

    def record(clip, prompt_key, mode, song_dir, audio_path, seconds, extra=None):
        info = sf.info(audio_path)
        with open(os.path.join(song_dir, "song.json")) as fh:
            song = json.load(fh)
        line = {"clip": clip, "prompt": prompt_key, "mode": mode, "lm": lm,
                "audio": os.path.relpath(audio_path, out_dir), "md5": md5(audio_path),
                "audio_s": round(info.duration, 3), "subtype": info.subtype, "s": round(seconds, 1),
                "bpm": song.get("bpm"), "keyscale": song.get("keyscale"),
                "instrumental": song.get("instrumental"), "seed": song.get("seed"),
                "caption": song.get("caption"), **(extra or {})}
        print("N5 " + json.dumps(line, ensure_ascii=False), flush=True)

    limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
    for clip, prompt_key, mode in jobs(lm)[:limit]:
        options = {"literal": True} if mode == "literal" else {}
        free["on"] = mode == "free"
        t = time.time()
        song = engine.generate(clip, PROMPTS[prompt_key], options, lambda _phase: None)
        free["on"] = False
        base_dir = song["dir"]
        base_audio = os.path.join(base_dir, song["audio"])
        if mode != "k1":
            record(clip, prompt_key, mode, base_dir, base_audio, time.time() - t)
            continue
        record(f"{clip}-orig", prompt_key, mode, base_dir, base_audio, time.time() - t)
        result, params = captured["result"], captured["params"]
        meta = result.extra_outputs.get("lm_metadata") or {}
        audio_params = result.audios[0]["params"]
        for shift in (1.0, 3.0):
            render = inference.GenerationParams(
                caption=meta.get("caption") or params.caption, lyrics=params.lyrics,
                instrumental=params.instrumental, bpm=params.bpm or params.cot_bpm,
                keyscale=params.keyscale or params.cot_keyscale or "",
                timesignature=params.timesignature or params.cot_timesignature or "",
                duration=params.duration,
                vocal_language=meta.get("vocal_language") or params.vocal_language,
                audio_codes=audio_params["audio_codes"], thinking=False, use_cot_metas=False,
                use_cot_caption=False, use_cot_language=False, enable_normalization=False,
                shift=shift,
            )
            config = inference.GenerationConfig(batch_size=1, use_random_seed=False,
                                                seeds=[song["seed"]], audio_format="wav32")
            render_dir = os.path.join(out_dir, f"{clip}-s{shift:g}")
            t = time.time()
            rendered = upstream_generate(engine.dit, engine.llm, render, config, save_dir=render_dir)
            seconds = time.time() - t
            if not rendered.success or not rendered.audios:
                raise RuntimeError(f"{clip} shift {shift} failed: {rendered.error}")
            path = rendered.audios[0]["path"]
            with open(os.path.join(render_dir, "song.json"), "w") as fh:
                json.dump({**{k: song[k] for k in ("prompt", "instrumental", "seed")},
                           "bpm": render.bpm, "keyscale": render.keyscale,
                           "caption": render.caption, "shift": shift}, fh, ensure_ascii=False)
            record(f"{clip}-s{shift:g}", prompt_key, mode, render_dir, path, seconds,
                   {"shift": shift, "same_as_orig": md5(path) == md5(base_audio),
                    "dit_seed": rendered.audios[0]["params"].get("seed")})


if __name__ == "__main__":
    main()
