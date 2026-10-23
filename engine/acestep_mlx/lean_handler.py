"""ACE-Step handler that never materializes the DiT decoder in torch on MPS.

Upstream loads the whole DiT in fp32 on MPS and then converts the decoder to MLX,
so the decoder exists twice and init peaks above 20 GB on a 16 GB Mac. Here the
torch model is loaded on CPU in the dtype upstream would use, its decoder parameters
are dropped, the MLX decoder is read from the checkpoint, and only the rest of the
torch model goes to MPS. LoRA is not supported on this path: upstream applies
adapters to the torch decoder.
"""

import gc
import os

import torch
from loguru import logger

from acestep.handler import AceStepHandler

from acestep_mlx.dit_weights import load_mlx_decoder


def drop_module_parameters(module: torch.nn.Module) -> int:
    """Replace every parameter of ``module`` with an empty tensor, keeping the structure.

    Args:
        module: The module whose weights are no longer needed.

    Returns:
        Number of parameter elements released.
    """
    released = 0
    with torch.no_grad():
        for param in module.parameters():
            released += param.numel()
            param.data = torch.empty(0, dtype=param.dtype, device=param.device)
    return released


class LeanAceStepHandler(AceStepHandler):
    """``AceStepHandler`` whose MPS init keeps a single, MLX-only DiT decoder."""

    def initialize_service(self, *args, use_mlx_dit: bool = True, **kwargs):
        """Refuse to disable MLX DiT: on MPS this handler has no torch decoder to fall back to.

        Raises:
            ValueError: If ``use_mlx_dit`` is False.
        """
        if not use_mlx_dit:
            raise ValueError("LeanAceStepHandler requires use_mlx_dit=True")
        return super().initialize_service(*args, use_mlx_dit=True, **kwargs)

    def _load_main_model_from_checkpoint(self, *, model_checkpoint_path, device, **kwargs) -> str:
        """Load the DiT so that the decoder lives only in MLX when running on MPS.

        Falls back to the upstream loader on any other device.
        """
        if device != "mps":
            return super()._load_main_model_from_checkpoint(
                model_checkpoint_path=model_checkpoint_path, device=device, **kwargs
            )
        saved = (self.offload_to_cpu, self.offload_dit_to_cpu)
        # With both offload flags set, upstream loads in self.dtype and keeps the
        # model on CPU. Loading in the final dtype (not bf16) matters: some buffers
        # (FSQ codebook) are computed at construction time, in the load dtype.
        self.offload_to_cpu, self.offload_dit_to_cpu = True, True
        try:
            attn = super()._load_main_model_from_checkpoint(
                model_checkpoint_path=model_checkpoint_path, device=device, **kwargs
            )
        finally:
            self.offload_to_cpu, self.offload_dit_to_cpu = saved

        released = drop_module_parameters(self.model.decoder)
        gc.collect()
        self.mlx_decoder = load_mlx_decoder(
            os.path.join(model_checkpoint_path, "model.safetensors"), self.config
        )
        self.use_mlx_dit = True
        # Already in self.dtype on CPU: a single copy lands on MPS, no intermediate.
        self.model = self.model.to(device)
        self.model.eval()
        self.silence_latent = self.silence_latent.to(device)
        logger.info(
            f"[lean] MLX decoder loaded from checkpoint; dropped {released / 1e9:.3f}B torch "
            f"decoder params; rest of the DiT on {device} as {self.dtype}"
        )
        return attn

    def _init_mlx_vae(self) -> bool:
        """Upstream MLX VAE init; ``ACESTEP_MLX_VAE_NOCOMPILE=1`` keeps decode uncompiled."""
        ok = super()._init_mlx_vae()
        if ok and os.environ.get("ACESTEP_MLX_VAE_NOCOMPILE") == "1":
            self._mlx_compiled_decode = self.mlx_vae.decode
            logger.info("[lean] MLX VAE decode left uncompiled")
        return ok

    def _resolve_mlx_decode_fn(self):
        """With ``ACESTEP_LEAN_MEMLOG=1``, wrap the decode callable to log MLX peak memory per tile."""
        import mlx.core as mx

        inner = super()._resolve_mlx_decode_fn()
        if os.environ.get("ACESTEP_LEAN_MEMLOG") != "1":
            return inner

        def _logged(z):
            mx.reset_peak_memory()
            out = inner(z)
            mx.eval(out)
            logger.info(
                f"[lean] VAE tile {tuple(z.shape)} -> {tuple(out.shape)}: "
                f"MLX peak {round(mx.get_peak_memory() / 2**20)} MB"
            )
            return out

        return _logged

    def _mlx_run_diffusion(self, *args, **kwargs):
        """Run upstream MLX diffusion; with ``ACESTEP_LEAN_MEMLOG=1`` log MLX memory around it."""
        import mlx.core as mx

        if os.environ.get("ACESTEP_LEAN_MEMLOG") != "1":
            return super()._mlx_run_diffusion(*args, **kwargs)
        mx.reset_peak_memory()
        result = super()._mlx_run_diffusion(*args, **kwargs)
        logger.info(
            f"[lean] after diffusion: MLX active {round(mx.get_active_memory() / 2**20)} MB, "
            f"cache {round(mx.get_cache_memory() / 2**20)} MB, "
            f"peak during diffusion {round(mx.get_peak_memory() / 2**20)} MB"
        )
        return result

    def _mlx_vae_decode(self, latents_torch):
        """Run upstream MLX VAE decode; with ``ACESTEP_LEAN_MEMLOG=1`` log MLX memory around it.

        With ``ACESTEP_MLX_CLEAR_BEFORE_VAE=1`` the MLX buffer cache left by the
        diffusion loop is released first (measured: no effect on the peak).
        """
        import mlx.core as mx

        if os.environ.get("ACESTEP_LEAN_MEMLOG") != "1":
            if os.environ.get("ACESTEP_MLX_CLEAR_BEFORE_VAE") == "1":
                mx.clear_cache()
            return super()._mlx_vae_decode(latents_torch)

        def _mb(x):
            return round(x / 2**20)

        if os.environ.get("ACESTEP_MLX_CLEAR_BEFORE_VAE") == "1":
            mx.clear_cache()
        logger.info(
            f"[lean] before VAE decode: MLX active {_mb(mx.get_active_memory())} MB, "
            f"cache {_mb(mx.get_cache_memory())} MB, peak so far {_mb(mx.get_peak_memory())} MB"
        )
        mx.reset_peak_memory()
        audio = super()._mlx_vae_decode(latents_torch)
        logger.info(
            f"[lean] after VAE decode: MLX active {_mb(mx.get_active_memory())} MB, "
            f"cache {_mb(mx.get_cache_memory())} MB, peak during decode {_mb(mx.get_peak_memory())} MB"
        )
        return audio

    def _init_mlx_dit(self, compile_model: bool = False) -> bool:
        """Keep the decoder built at load time instead of converting it from torch."""
        if getattr(self, "mlx_decoder", None) is not None and self.use_mlx_dit:
            self.mlx_dit_compiled = compile_model
            return True
        return super()._init_mlx_dit(compile_model=compile_model)
