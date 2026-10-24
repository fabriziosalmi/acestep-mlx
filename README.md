# acestep-mlx

Working name. A native macOS app that turns a sentence into a song, built on
[ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5), with MLX used wherever
possible and zero user-facing options.

Status: research prototype, run from a development install on a Mac mini M4 16 GB. The app
works end to end (a sentence in, a song out, a history to play from); it is not bundled with
its engine yet and has not been measured on the laptop.

## Layout

- `engine/acestep_mlx/`: lean engine, a subclass of upstream's handler, the recipe and the
  JSON-lines service the app talks to (tests in `engine/tests/`, 45 with `unittest`).
- `app/`: SwiftUI app as a SwiftPM package: `AceStepKit` (engine process, state, history),
  `AceStep` (the window), `AceStepE2E` (drives the real engine), 30 Swift Testing tests.
- `bench/`: measurement harness per step (`f0`, `n3` to `n7`), each with a prediction written
  before its run and the result appended after it (`prediction_*.txt`).
- `bench/results/<date>-<host>/`: raw logs, frozen inputs, audio checksums, listening-test keys and votes.

## How a song is made

The app starts `python -m acestep_mlx.engine_service --project <upstream> --out <songs> --preset <name>`
with upstream's venv. The engine loads the DiT and the 5 Hz LM once and serves one request per
line on stdin; stdout carries protocol events only (`ready`, `phase`, `done`, `error`).

The recipe (`engine/acestep_mlx/recipe.py`, `prompt_rules.py`):

- The LM rewrites the prompt into a caption, lyrics and metadata. Option `literal` passes the
  prompt to the DiT as written.
- A prompt that names no voice (Italian or English word list, negations win) is instrumental.
- A BPM (30-300) or a key written in the prompt constrains the LM; without it the LM chooses.
- Duration is only an explicit option (`duration`), never read from the prompt; default 30 s.
- Output is 32-bit float WAV left raw, or a 320 kbps MP3 (`format: "mp3"`) peak-normalized to
  -1 dBFS, encoded by libsndfile (no ffmpeg).
- `like: <song folder>` makes another version of a song: same text and metadata, new seed.
- The seed is drawn by the engine and passed to both the LM and the DiT, so the one recorded in
  `song.json` reproduces the song (upstream records the LM's seed and lets the DiT draw another).

Each song is a folder `<songs>/YYYYmmdd-HHMMSS-<id>/` with the audio and `song.json`.

Presets are chosen by the app: `laptop` on a machine with an internal battery, `mini`
otherwise (`ACESTEP_PRESET` overrides). Both use LM 0.6B and DiT `acestep-v15-turbo` at shift
1.0. `mini` allows up to 240 s; `laptop` keeps 30 s as a precaution until it is measured on
the laptop. The app stops an idle engine to give its memory back, after 1200 s on `mini` and
120 s on `laptop` (`ACESTEP_IDLE_SECONDS` overrides), and starts it again for the next song.

## Run it (development install)

On a Mac with `bench/f0/setup.sh` done (upstream in `~/acestep-f0/ACE-Step-1.5` with its
checkpoints) and this repository in `~/acestep-mlx`:

```sh
cd ~/acestep-mlx/app
./scripts/bundle.sh            # builds build/ACE-Step.app, signed ad hoc
open build/ACE-Step.app
swift run AceStepE2E           # end-to-end check against the real engine
```

`ACESTEP_PYTHON`, `ACESTEP_ENGINE` and `ACESTEP_PROJECT` override the default paths. Songs go
to `~/Music/ACE-Step`, the engine log to `~/Library/Logs/ACE-Step/engine.log`.

## Results so far (2026-10-23, Mac mini M4 16 GB, macOS 26.4.1)

Upstream `ace-step/ACE-Step-1.5` at `ca1e85f`, `acestep-v15-turbo` + `acestep-5Hz-lm-0.6B`,
MLX backend, batch 1, 30 s, seed 42. Every prediction was written before its run
(`prediction*.txt`).

| Run | Change | Footprint while generating (MB, `footprint`) | Generation | Audio vs `ref1` |
|---|---|---|---|---|
| `run1` (F0) | upstream defaults | 21504 | MPS out of memory | none |
| `f1` | torch decoder emptied after MLX conversion, MLX decoder bf16 | 13312 | 42.4 s (incl. LM codes) | not comparable (LM sampling) |
| `ref1`, `ref2` | replay, MLX decoder fp32 | 14336 | 60-73 s | bit-identical to each other |
| `a` | replay, MLX decoder bf16 | 11264 | 29.4 s | bit-identical |
| `bv` | `a` + MLX VAE fp16 | 11264 | 32.0 s | SNR 58.1 dB |
| `bt` | `a` + torch model and text encoder bf16 | 7528 | 23.1 s | SNR 16.8 dB, corr 0.990 |
| `b` | `bt` + `bv` | 7214 | 21.8 s | SNR 16.8 dB, corr 0.990 |
| `c` | `a` + CPU round-trip of torch modules | 11264 | 18.0 s | bit-identical |

Findings:

1. Upstream holds the DiT decoder twice in fp32 on MPS (torch and MLX copies) and does not fit 16 GB.
2. The checkpoint is stored in BF16, so keeping weights in bf16 is lossless; `a` is bit-identical to the fp32 reference.
3. Computing in bf16 (torch) changes the output; the MLX VAE in fp16 is close to transparent.
4. Every slimmed run still peaks at 20.5-21.9 GB (`/usr/bin/time -l` peak memory footprint, 23.05 GB for F0), because the torch model is materialized in fp32 before conversion.
5. A CPU round-trip does not make the MPS allocator release memory (`c`).

Generation times vary with swap state and are indicative only.

## Lean engine (same host and inputs)

`engine/acestep_mlx/lean_handler.py` subclasses upstream's handler. Replays with frozen
inputs; "audio" compares against the upstream fp32 path (`ref1`, md5 `4e731ce4...`).

| Change | Process peak (`time -l`) | Audio vs `ref1` |
|---|---|---|
| Upstream path, torch decoder emptied after conversion (`a`) | 20.54 GB | bit-identical |
| MLX decoder read from safetensors, torch decoder never on MPS | 20.28-20.49 GB (init 11-15 s instead of 48-74 s) | bit-identical |
| + VAE tiles 64 frames, overlap 16 | 17.84 GB | bit-identical |
| + torch VAE weights dropped after MLX VAE init | 17.80 GB | bit-identical |
| + MLX buffer cache limited to 64 MB | 14.09-14.14 GB (3 runs) | bit-identical |
| + text encoder on MLX (N2) | 13.22 GB | SNR 79.8 dB |
| + condition encoder, audio tokenizer, detokenizer on MLX (N3) | 10.48 GB (2 runs) | SNR 77.7 dB |

From N2 on, the MLX modules keep the stored bf16 weights and compute in fp32. Their outputs
differ from torch MPS by less than torch CPU differs from torch MPS on the same inputs
(N3 encoder: max abs 5.0e-6 vs 7.1e-6; `bench/n3/noise_floor.py`). After N3, torch holds
4 MB on MPS and no torch weights are used for compute except the 2048-value null condition.

Full describe-to-song flow (LM 0.6B on MLX, `create_sample`, thinking on, 30 s), peak and
time from process start:

| Engine state | Peak | Time | Swap growth |
|---|---|---|---|
| N1 | 15.87 GB | 75 s | |
| N2 | 14.20 GB | 63.9 s | +0.9 GB |
| N3 | 11.78 GB | 59.6 s (init 8.2, LM init 20.1, lyrics and metadata 9.4, generation 19.4) | +21 MB |

Loading the torch model in bf16 is not lossless: FSQ codebook buffers are built in the load dtype.
Each step's written prediction and result are in `bench/n*/prediction_n*.txt`.

## Duration and LM size (N4, one process per song)

| LM | Song length | Peak (`time -l`) | Song time |
|---|---|---|---|
| 0.6B | 30 s | 11.78 GB | 21.8 s |
| 0.6B | 60 s | 11.79 GB | 34.7 s |
| 0.6B | 120 s | 11.87 GB | 66.7 s |
| 0.6B | 240 s | 12.21 GB | 130.1 s |
| 0.6B | 240 s, vocal | 13.40 GB | 141.6 s |
| 1.7B | 30 s | 13.97 GB | 41.6 s |
| 1.7B | 240 s | 14.15 GB | 202.7 s |

Memory barely grows with duration: the peak is the render phase. Time is linear, about 0.16 s
of LM codes and 0.37 s of rendering per second of music. The vocal run's extra 1.19 GB is outside
MLX and not attributed.

## Blind listening tests (N5, N6)

One listener, pairs of 30 s clips with the same prompt, level-matched, order and sides shuffled,
the key kept off the listening machine until the votes were in. Each round's decision rule was
written before any clip was made (`bench/n5/prediction_n5.txt`, `bench/n6/prediction_n6.txt`);
none of them was met, so every default stayed.

| Knob | Result | Decision |
|---|---|---|
| shift 3.0 vs 1.0 (same LM codes and seed) | 1 win for 3.0, 3 ties | keep 1.0 |
| LM 1.7B vs 0.6B | 1 win for 0.6B, 3 ties | keep 0.6B |
| BPM/key forced vs free | 3 of 3 for forced, none broken | keep forced |
| literal vs rewrite, two rounds | literal 8 of 12, rewrite 4; broken flags 0 vs 2 | keep rewrite, `literal` stays an option |

Without the constraint the LM ignored the BPM written in the prompt in 3 of 3 runs (128 -> 100,
124 -> 50, 180 -> 100). Freetekno at 180 BPM is a declared limit: 5 of 8 clips were judged broken.
Both pairs made of the same file twice were judged equal; a pair of two runs of the same
settings got a preference, so a single pair's vote is at seed-noise level.

## Memory at rest (N7)

An engine with both models loaded holds 8.9 GB (`footprint`, 8875-8880 MB in 3 cycles). The app
stops the process after the idle period; it was gone within 1 s each time, and the next request
waited 33 s for the load, 53-54 s to a 30 s song. The system's free+inactive counter rose by
4.6-4.7 GB at each stop, less than the footprint; the gap is not attributed.

## Reproduce

On the measurement host, `bench/f0/setup.sh` clones upstream into `~/acestep-f0`, syncs
dependencies and downloads the four checkpoints. `bench/f0/run_f0.sh <tag>` runs one
measurement under `/usr/bin/time -l` with a 2 s swap sampler. Environment switches read by
`f0_measure.py`:

| Variable | Effect |
|---|---|
| `F0_SLIM=1` | empty the torch decoder parameters after MLX conversion |
| `F0_MLX_BF16=1` | cast the MLX decoder to bf16 |
| `F0_TORCH_BF16=1` | move torch model and text encoder to bf16 (via CPU) |
| `F0_COMPACT=1` | CPU round-trip of torch modules without a dtype change |
| `F0_SAVE_FIXED=path` | save the conditioning used by a run |
| `F0_REPLAY=path` | replay saved conditioning with the LM skipped |
| `F0_TAG=name` | output subdirectory |

`bench/f0/compare.py ref tag...` prints SNR, correlation and log-spectral distance.
