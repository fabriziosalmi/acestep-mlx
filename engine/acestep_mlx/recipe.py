"""The engine's recipe: presets, request options, and the plan they produce for one song.

Defaults decided 2026-10-23: the LM may rewrite the prompt (``literal`` opts out);
BPM and key written in the prompt bind the LM; a prompt that names no voice is
instrumental; duration is only ever an explicit option, never read from the prompt;
output is 32-bit float WAV left raw, or a 320 kbps MP3 peak-normalized to
-1 dBFS (MP3 would clip otherwise; see audio_export.py). ``plan_request`` is pure,
so every rule is unit-tested.
"""

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from acestep_mlx.prompt_rules import explicit_bpm, explicit_key, wants_vocals

DURATION_MIN = 10.0  # upstream acestep/constants.py


@dataclass(frozen=True)
class Preset:
    """Machine-level choices made once at engine start."""

    name: str
    lm_model: str
    default_duration: float
    max_duration: float


# max_duration values are provisional until the duration series (bench/n4) is measured.
PRESETS = {
    "mini": Preset("mini", "acestep-5Hz-lm-0.6B", default_duration=30.0, max_duration=240.0),
    "laptop": Preset("laptop", "acestep-5Hz-lm-0.6B", default_duration=30.0, max_duration=30.0),
}
# Upstream always writes this; MP3 is made from it by audio_export.
UPSTREAM_FORMAT = "wav32"
FORMATS = ("wav", "mp3")
_SONG_NAME = re.compile(r"^\d{8}-\d{6}-[0-9a-z-]{1,8}$")


@dataclass(frozen=True)
class Options:
    """Per-request options; every field has the default the app uses today."""

    literal: bool = False
    vocals: Optional[bool] = None  # None: decided from the prompt
    duration: Optional[float] = None  # None: the preset's default
    format: str = "wav"
    like: Optional[str] = None  # folder name of a song to make another version of


@dataclass(frozen=True)
class Plan:
    """What the engine will do for one request; recorded in song.json."""

    literal: bool
    instrumental: bool
    bpm: Optional[int]  # forced from the prompt, else None (the LM chooses)
    keyscale: Optional[str]
    duration: float
    format: str  # "wav" (raw float) or "mp3" (normalized, see audio_export)
    like: Optional[str] = None
    sources: Dict[str, str] = field(default_factory=dict)


def parse_options(raw: Optional[Dict[str, Any]], preset: Preset) -> Options:
    """Validate the request's ``options`` object.

    Raises:
        ValueError: For unknown keys, wrong types, a duration outside
            ``[DURATION_MIN, preset.max_duration]``, an unknown format or a bad song name.
    """
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("options must be an object")
    unknown = set(raw) - set(Options.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown options {sorted(unknown)}")
    opts = Options(**raw)
    if not isinstance(opts.literal, bool):
        raise ValueError("literal must be true or false")
    if opts.vocals is not None and not isinstance(opts.vocals, bool):
        raise ValueError("vocals must be true, false or null")
    if opts.duration is not None:
        if isinstance(opts.duration, bool) or not isinstance(opts.duration, (int, float)):
            raise ValueError("duration must be a number of seconds")
        if not DURATION_MIN <= opts.duration <= preset.max_duration:
            raise ValueError(f"duration must be between {DURATION_MIN:g} and "
                             f"{preset.max_duration:g} s on the {preset.name} preset")
    if opts.format not in FORMATS:
        raise ValueError(f"format must be one of {sorted(FORMATS)}")
    if opts.like is not None and not (isinstance(opts.like, str) and _SONG_NAME.match(opts.like)):
        raise ValueError("like must be the folder name of a song")
    return opts


def plan_request(prompt: str, opts: Options, preset: Preset) -> Plan:
    """Apply the default rules to a prompt and its options."""
    bpm, keyscale = explicit_bpm(prompt), explicit_key(prompt)
    vocals = wants_vocals(prompt) if opts.vocals is None else opts.vocals
    return Plan(
        literal=opts.literal, instrumental=not vocals, bpm=bpm, keyscale=keyscale,
        duration=float(opts.duration if opts.duration is not None else preset.default_duration),
        format=opts.format, like=opts.like,
        sources={
            "vocals": "prompt" if opts.vocals is None else "option",
            "bpm": "prompt" if bpm is not None else "lm",
            "keyscale": "prompt" if keyscale is not None else "lm",
        },
    )


def song_folder(out_dir: str, name: str) -> str:
    """Absolute folder of an existing song named ``name`` inside ``out_dir``.

    Raises:
        ValueError: If the name is malformed or the folder is not a song in ``out_dir``.
    """
    if not _SONG_NAME.match(name):
        raise ValueError("like must be the folder name of a song")
    root = os.path.realpath(out_dir)
    folder = os.path.realpath(os.path.join(root, name))
    if os.path.dirname(folder) != root or not os.path.isfile(os.path.join(folder, "song.json")):
        raise ValueError(f"no song named {name}")
    return folder
