"""ACE-Step handler that never materializes the DiT decoder in torch on MPS.

Upstream loads the whole DiT in fp32 on MPS and then converts the decoder to MLX,
so the decoder exists twice and init peaks above 20 GB on a 16 GB Mac. Here the
torch model is loaded on CPU in the dtype upstream would use, its decoder parameters
are dropped, the MLX decoder is read from the checkpoint, and only the rest of the
torch model goes to MPS. LoRA is not supported on this path: upstream applies
adapters to the torch decoder. The condition encoder, audio tokenizer and detokenizer
are replaced by MLX bridges before the move (``condition.py``).
"""

import gc
import os

import torch
from loguru import logger

from acestep.handler import AceStepHandler

from acestep_mlx.condition import MLXConditionMixin
from acestep_mlx.dit_weights import load_mlx_decoder
from acestep_mlx.memlog import MemLogMixin
from acestep_mlx.text_encoder import MLXTextEncoderMixin
from acestep_mlx.vae_tiling import decode_tiled

DEFAULT_VAE_CHUNK = 64
DEFAULT_VAE_OVERLAP = 16
DEFAULT_MLX_CACHE_LIMIT_MB = 64


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


class LeanAceStepHandler(MLXConditionMixin, MLXTextEncoderMixin, MemLogMixin, AceStepHandler):
    """``AceStepHandler`` whose MPS init keeps a single, MLX-only DiT decoder."""

    def initialize_service(self, *args, use_mlx_dit: bool = True, **kwargs):
        """Refuse to disable MLX DiT: on MPS this handler has no torch decoder to fall back to.

        Raises:
            ValueError: If ``use_mlx_dit`` is False.
        """
        if not use_mlx_dit:
            raise ValueError("LeanAceStepHandler requires use_mlx_dit=True")
        # MLX keeps freed buffers for reuse and the process footprint counts them.
        # Measured on the Mac mini: 64 MB gives the lowest peak (14.09-14.14 GB over
        # three runs vs 17.80 GB unlimited) and the fastest generation; 0 MB is slower.
        limit_mb = os.environ.get(
            "ACESTEP_LEAN_MLX_CACHE_LIMIT_MB", str(DEFAULT_MLX_CACHE_LIMIT_MB)
        )
        if limit_mb != "unlimited":
            import mlx.core as mx

            mx.set_cache_limit(int(limit_mb) * 2**20)
            logger.info(f"[lean] MLX buffer cache limited to {limit_mb} MB")
        return super().initialize_service(*args, use_mlx_dit=True, **kwargs)

    def _ensure_models_present(self, *, checkpoint_path, config_path, prefer_source=None,
                               vae_variant=None):
        """Check the DiT, VAE and text encoder folders without downloading anything.

        The engine fetches exactly the files it needs before init (``models.ensure``).
        Upstream's check wants every component of its main repository, including the 1.7B
        LM this engine does not use, and downloads the whole repository when one is missing.

        Returns:
            None when every folder has weights, else upstream's (message, False) error.
        """
        from acestep.model_downloader import check_model_exists

        missing = [name for name in (config_path, "vae", "Qwen3-Embedding-0.6B")
                   if not check_model_exists(name, checkpoint_path)]
        if missing:
            return f"ERROR: model weights missing in {checkpoint_path}: {', '.join(missing)}", False
        return None

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
        checkpoint_file = os.path.join(model_checkpoint_path, "model.safetensors")
        self.mlx_decoder = load_mlx_decoder(checkpoint_file, self.config)
        self._install_mlx_condition(checkpoint_file)
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
        """Upstream MLX VAE init, then drop the torch VAE weights it was converted from."""
        ok = super()._init_mlx_vae()
        if ok and os.environ.get("ACESTEP_LEAN_KEEP_TORCH_VAE") != "1":
            # Text-to-music decodes with the MLX VAE; the torch copy is only read for
            # its dtype, which empty parameters keep. Source/reference-audio encode
            # paths would fail loudly, not fall back.
            released = drop_module_parameters(self.vae)
            gc.collect()
            torch.mps.empty_cache()
            logger.info(f"[lean] dropped {released / 1e6:.1f}M torch VAE params after MLX VAE init")
        return ok

    def _mlx_decode_single(self, z_nlc, decode_fn=None):
        """Tile the VAE decode in 64-frame tiles with 16 frames of overlap.

        Measured on the Mac mini: output bit-identical to upstream's 256/64 tiling,
        process peak 17.8 GB instead of 20.5 GB. Overlap 12 already changes the
        output (SNR 130.8 dB), so 16 is the smallest exact value found.
        ``ACESTEP_LEAN_VAE_CHUNK``/``_OVERLAP`` override; ``ACESTEP_LEAN_VAE_CHUNK=upstream``
        restores upstream's tiling.
        """
        chunk = os.environ.get("ACESTEP_LEAN_VAE_CHUNK", str(DEFAULT_VAE_CHUNK))
        if chunk == "upstream":
            return super()._mlx_decode_single(z_nlc, decode_fn=decode_fn)
        overlap = int(os.environ.get("ACESTEP_LEAN_VAE_OVERLAP", str(DEFAULT_VAE_OVERLAP)))
        return decode_tiled(z_nlc, decode_fn or self._resolve_mlx_decode_fn(), int(chunk), overlap)

    def _init_mlx_dit(self, compile_model: bool = False) -> bool:
        """Keep the decoder built at load time instead of converting it from torch."""
        if getattr(self, "mlx_decoder", None) is not None and self.use_mlx_dit:
            self.mlx_dit_compiled = compile_model
            return True
        return super()._init_mlx_dit(compile_model=compile_model)
