# acestep-mlx

Working name. A native macOS app that turns a sentence into a song, built on
[ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5), with MLX used wherever
possible and zero user-facing options.

Status: research. No app yet. This repository holds the measurement harness, the raw
results it produced, and a lean engine that runs upstream's pipeline on MLX.

## Layout

- `bench/f0/`: measurement harness (setup, one-shot run, frozen-input replay, audio comparison).
- `bench/results/<date>-<host>/`: raw logs, written predictions, frozen inputs and audio checksums.
- `engine/acestep_mlx/`: lean engine, a subclass of upstream's handler (tests in `engine/tests/`).
- `app/`: SwiftUI app (not started).

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
