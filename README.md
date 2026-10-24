# acestep-mlx

Working name. A native macOS app that turns a sentence into a song, built on
[ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5), with MLX used wherever possible and
no options in the window.

Version 0.1.0: a research prototype for Apple Silicon Macs with 16 GB of memory. It runs from a
development install (upstream's checkout and venv next to this repository); the engine is not
bundled inside the app yet.

## What it does

Type a description, press Make song (Cmd-Return), and the song appears in the history, where it
can be played, shown in Finder, have its lyrics copied, or be made again in another version.
The window always says what the engine is doing: loading models, writing lyrics, composing,
rendering, or that the models are unloaded to free memory.

## Measured on a Mac mini M4 16 GB (macOS 26.4.1)

| | Result |
|---|---|
| Engine start (DiT, LM 0.6B, text encoder) | 33-34 s |
| 30 s song | 21.8 s, peak 11.78 GB |
| 240 s song | 130 s, peak 12.21 GB (13.40 GB with vocals) |
| Engine at rest with models loaded | 8.9 GB |
| After the idle stop | the engine process is gone within 1 s |
| Next song after the idle stop | 33 s to load, 53-54 s to a 30 s song |
| Audio vs upstream's fp32 path, same inputs | SNR 77.7 dB |

Peaks are the `/usr/bin/time -l` peak memory footprint of the engine process. Song time grows
linearly, about 0.5 s per second of music; memory barely grows with duration. Upstream with its
defaults does not fit in 16 GB on this machine (MPS out of memory).

## How a song is made

The app runs `python -m acestep_mlx.engine_service` with upstream's venv. The engine loads the
models once and serves one request per line on stdin; stdout carries protocol events only
(`ready`, `phase`, `done`, `error`). DiT, VAE, text encoder, condition encoder, audio tokenizer
and detokenizer run on MLX with the stored bf16 weights and fp32 compute; torch is left as
upstream's orchestration code. The 5 Hz LM runs on upstream's MLX backend.

Defaults (`engine/acestep_mlx/recipe.py`, `prompt_rules.py`):

- The LM turns the prompt into a caption, lyrics and metadata.
- A prompt that names no voice (Italian or English word list; "senza voce", "no vocals" and the
  like win) makes an instrumental.
- A BPM (30-300) or a key written in the prompt constrains the LM; otherwise the LM chooses.
- Songs are 30 s; duration is never read from the prompt.
- DiT `acestep-v15-turbo`, 8 steps, shift 1.0; LM `acestep-5Hz-lm-0.6B`.
- The seed is drawn by the engine and used by both the LM and the DiT, so the seed in
  `song.json` reproduces the song.

These defaults were kept by blind A/B listening tests, with each decision rule written before
any clip was made (`bench/n5`, `bench/n6`): shift 3.0, LM 1.7B and the literal prompt did not
win clearly; the forced BPM won 3 of 3.

Request options, not shown in the window: `literal` (the prompt goes to the DiT as written),
`vocals` (true or false, instead of the word list), `duration` (10 s up to the preset's maximum),
`format` (`wav`: 32-bit float, raw; `mp3`: 320 kbps, peak at -1 dBFS, encoded by libsndfile),
and `like` (a song folder: same text and metadata, new seed; the window's "Another like this").

Each song is a folder `~/Music/ACE-Step/YYYYmmdd-HHMMSS-<id>/` with the audio and `song.json`
(prompt, caption, lyrics, metadata, seed, models, render time).

## Presets

The app picks `laptop` on a Mac with an internal battery and `mini` otherwise
(`ACESTEP_PRESET` overrides).

| Preset | LM | Longest song | Idle stop |
|---|---|---|---|
| `mini` | 0.6B | 240 s | after 20 min |
| `laptop` | 0.6B | 30 s | after 2 min |

The laptop preset has not been measured on a laptop; its 30 s cap is a precaution.
`ACESTEP_IDLE_SECONDS` overrides the idle period.

## Known limits

- Freetekno at 180 BPM often comes out broken (5 of 8 clips in the listening tests).
- Without a BPM in the prompt the 0.6B LM can pick odd tempos (40-50 BPM for an Italian folk
  ballad in 3 of 6 runs), and it does not always follow the voice asked for.
- A song with vocals peaks about 1.2 GB higher than an instrumental one; not attributed yet.

## Run it

Requirements: Apple Silicon, macOS 14 or later, about 9 GB of disk (7.2 GB of checkpoints,
1.3 GB of venv), Xcode or the Command Line Tools, and [uv](https://github.com/astral-sh/uv).

```sh
git clone https://github.com/ace-step/ACE-Step-1.5 ~/acestep-f0/ACE-Step-1.5
git -C ~/acestep-f0/ACE-Step-1.5 checkout ca1e85f
git clone https://github.com/fabriziosalmi/acestep-mlx ~/acestep-mlx
~/acestep-mlx/bench/f0/setup.sh        # uv sync, then the turbo DiT, VAE, text encoder and LM 0.6B
cd ~/acestep-mlx/app
./scripts/bundle.sh                    # builds build/ACE-Step.app, signed ad hoc
open build/ACE-Step.app
```

`ACESTEP_PYTHON`, `ACESTEP_ENGINE` and `ACESTEP_PROJECT` override the default paths. The engine
log is `~/Library/Logs/ACE-Step/engine.log`.

Checks:

```sh
cd ~/acestep-mlx/engine && ~/acestep-f0/ACE-Step-1.5/.venv/bin/python -m unittest discover -s tests
cd ~/acestep-mlx/app && swift test             # needs Xcode (Swift Testing)
cd ~/acestep-mlx/app && swift run AceStepE2E   # one real song through the app's model
```

## Layout

- `engine/acestep_mlx/`: the lean engine (a subclass of upstream's handler), the MLX modules, the
  recipe and the JSON-lines service; tests in `engine/tests/`.
- `app/`: SwiftPM package. `AceStepKit` (engine process, state, history), `AceStep` (the
  window), `AceStepE2E` (end-to-end check), `scripts/bundle.sh`.
- `bench/`: one folder per measured step, each with its prediction written before the run and
  the result after it; raw logs, listening-test keys and votes in `bench/results/`.
