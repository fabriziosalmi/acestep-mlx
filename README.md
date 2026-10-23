# acestep-mlx

Working name. A native macOS app that turns a sentence into a song, built on
[ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5), with MLX used wherever
possible and zero user-facing options.

Status: research. No app yet. This repository currently holds the measurement
harness and the raw results it produced.

## Layout

- `bench/f0/`: measurement harness (setup, one-shot run, frozen-input replay, audio comparison).
- `bench/results/<date>-<host>/`: raw logs, written predictions, frozen inputs and audio checksums.
- `engine/`: MLX engine (not started).
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
