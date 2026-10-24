"""Opt-in MLX memory logging around diffusion and VAE tiles (``ACESTEP_LEAN_MEMLOG=1``).

Used by the bench to locate the generation peak. Off by default, in which case
every method defers to the next class in the MRO unchanged.
"""

import os

from loguru import logger


def _enabled() -> bool:
    """Return True when memory logging is requested."""
    return os.environ.get("ACESTEP_LEAN_MEMLOG") == "1"


def _mb(value: int) -> int:
    """Bytes to whole MiB."""
    return round(value / 2**20)


class MemLogMixin:
    """Wraps upstream MLX diffusion and VAE decode with MLX memory reports."""

    def _resolve_mlx_decode_fn(self):
        """Wrap the decode callable to log the MLX peak per VAE tile."""
        import mlx.core as mx

        inner = super()._resolve_mlx_decode_fn()
        if not _enabled():
            return inner

        def _logged(z):
            mx.reset_peak_memory()
            out = inner(z)
            mx.eval(out)
            logger.info(f"[lean] VAE tile {tuple(z.shape)} -> {tuple(out.shape)}: "
                        f"MLX peak {_mb(mx.get_peak_memory())} MB")
            return out

        return _logged

    def _mlx_run_diffusion(self, *args, **kwargs):
        """Log MLX active, cache and peak memory after the diffusion loop."""
        import mlx.core as mx

        if not _enabled():
            return super()._mlx_run_diffusion(*args, **kwargs)
        mx.reset_peak_memory()
        result = super()._mlx_run_diffusion(*args, **kwargs)
        logger.info(f"[lean] after diffusion: MLX active {_mb(mx.get_active_memory())} MB, "
                    f"cache {_mb(mx.get_cache_memory())} MB, "
                    f"peak during diffusion {_mb(mx.get_peak_memory())} MB")
        return result

    def _mlx_vae_decode(self, latents_torch):
        """Log MLX memory before and after the whole VAE decode."""
        import mlx.core as mx

        if not _enabled():
            return super()._mlx_vae_decode(latents_torch)
        logger.info(f"[lean] before VAE decode: MLX active {_mb(mx.get_active_memory())} MB, "
                    f"cache {_mb(mx.get_cache_memory())} MB, "
                    f"peak so far {_mb(mx.get_peak_memory())} MB")
        mx.reset_peak_memory()
        audio = super()._mlx_vae_decode(latents_torch)
        logger.info(f"[lean] after VAE decode: MLX active {_mb(mx.get_active_memory())} MB, "
                    f"cache {_mb(mx.get_cache_memory())} MB, "
                    f"peak during decode {_mb(mx.get_peak_memory())} MB")
        return audio
