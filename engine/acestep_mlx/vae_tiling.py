"""Overlap-discard tiling for the MLX VAE decode with a configurable tile and overlap.

Upstream (``mlx_vae_decode_native.py``) fixes the overlap at 64 latent frames and
the tile at 192 or more. The decoder's receptive field is much smaller, so smaller
tiles keep the output identical while cutting the per-tile memory, which grows with
the tile length. The trimming logic below is upstream's, parameterized.
"""

import math
from typing import Callable, List, Tuple

Window = Tuple[int, int, int, int]  # core_start, core_end, win_start, win_end


def tile_plan(latent_frames: int, chunk: int, overlap: int) -> List[Window]:
    """Return the tiles upstream's loop would visit for these sizes.

    Args:
        latent_frames: Number of latent frames to decode.
        chunk: Tile length in latent frames, overlap included.
        overlap: Frames decoded on each side of a tile's core and then discarded.

    Returns:
        One ``(core_start, core_end, win_start, win_end)`` per tile, or a single
        whole-input window when the input fits in one tile.

    Raises:
        ValueError: If ``chunk <= 2 * overlap`` (no core left) or sizes are negative.
    """
    if overlap < 0 or chunk <= 2 * overlap:
        raise ValueError(f"chunk ({chunk}) must exceed 2 * overlap ({overlap})")
    if latent_frames <= chunk:
        return [(0, latent_frames, 0, latent_frames)]
    stride = chunk - 2 * overlap
    plan = []
    for idx in range(math.ceil(latent_frames / stride)):
        core_start = idx * stride
        core_end = min(core_start + stride, latent_frames)
        plan.append((core_start, core_end, max(0, core_start - overlap),
                     min(latent_frames, core_end + overlap)))
    return plan


def decode_tiled(z_nlc, decode_fn: Callable, chunk: int, overlap: int):
    """Decode ``[1, frames, C]`` MLX latents tile by tile and stitch the cores.

    Args:
        z_nlc: MLX latents in NLC layout.
        decode_fn: Callable decoding a latent window to ``[1, samples, channels]``.
        chunk: Tile length in latent frames.
        overlap: Discarded frames on each side.

    Returns:
        The decoded audio, ``[1, samples, channels]``.
    """
    import mlx.core as mx

    plan = tile_plan(z_nlc.shape[1], chunk, overlap)
    if len(plan) == 1:
        return decode_fn(z_nlc)
    parts, upsample = [], None
    for idx, (core_start, core_end, win_start, win_end) in enumerate(plan):
        audio = decode_fn(z_nlc[:, win_start:win_end, :])
        mx.eval(audio)
        if upsample is None:
            upsample = audio.shape[1] / (win_end - win_start)
        trim_start = int(round((core_start - win_start) * upsample))
        trim_end = int(round((win_end - core_end) * upsample))
        end_idx = audio.shape[1] - trim_end if trim_end > 0 else audio.shape[1]
        part = audio[:, trim_start:end_idx, :]
        mx.eval(part)
        parts.append(part)
        del audio
        if (idx + 1) % 4 == 0:
            mx.clear_cache()
    return mx.concatenate(parts, axis=1)
