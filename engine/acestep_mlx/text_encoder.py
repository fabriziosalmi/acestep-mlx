"""Qwen3-Embedding text encoder on MLX, replacing upstream's torch copy.

Weights stay in their stored bf16; embeddings are cast to fp32 before the layers,
so MLX promotes every matmul and norm to fp32, matching upstream's fp32 compute on
MPS. ``ACESTEP_LEAN_TEXT_ENCODER`` selects ``mlx`` (default; measured on the Mac mini:
audio SNR 79.8 dB vs upstream, min token cosine 0.99999988), ``torch`` (upstream), or
``compare`` (MLX outputs, with the torch encoder kept to log the difference).
"""

import json
import os

import numpy as np
import torch
from loguru import logger

from acestep_mlx.diffs import diff_report


class MLXTextEncoder:
    """Qwen3 hidden states and token embeddings computed with mlx-lm's Qwen3Model."""

    def __init__(self, model):
        """Wrap an evaluated ``mlx_lm.models.qwen3.Qwen3Model``."""
        self.model = model

    @classmethod
    def load(cls, path: str) -> "MLXTextEncoder":
        """Build the encoder from a Qwen3-Embedding checkpoint directory.

        Raises:
            ValueError: If checkpoint keys do not match the model (strict load).
        """
        import mlx.core as mx
        from mlx_lm.models.qwen3 import ModelArgs, Qwen3Model

        with open(os.path.join(path, "config.json")) as fh:
            args = ModelArgs.from_dict(json.load(fh))
        model = Qwen3Model(args)
        weights = mx.load(os.path.join(path, "model.safetensors"))
        model.load_weights(list(weights.items()), strict=True)
        mx.eval(model.parameters())
        return cls(model)

    def embed(self, ids: np.ndarray) -> np.ndarray:
        """Token embeddings as float32, ``[B, L]`` ids to ``[B, L, H]``."""
        import mlx.core as mx

        out = self.model.embed_tokens(mx.array(ids)).astype(mx.float32)
        return np.array(out)

    def hidden_states(self, ids: np.ndarray) -> np.ndarray:
        """Final-norm hidden states as float32 (HF ``last_hidden_state``), causal mask."""
        import mlx.core as mx

        tokens = mx.array(ids)
        h = self.model(tokens, input_embeddings=self.model.embed_tokens(tokens).astype(mx.float32))
        return np.array(h)


class MLXTextEncoderMixin:
    """Handler mixin: tokenizer from upstream, embeddings from MLX."""

    def _load_text_encoder_and_tokenizer(self, *, checkpoint_dir: str, device: str) -> str:
        """Load the tokenizer and the MLX encoder; the torch encoder only in compare mode."""
        from transformers import AutoTokenizer

        self._te_mode = os.environ.get("ACESTEP_LEAN_TEXT_ENCODER", "mlx")
        if self._te_mode not in ("torch", "mlx", "compare"):
            raise ValueError(
                f"ACESTEP_LEAN_TEXT_ENCODER must be torch, mlx or compare, not {self._te_mode}"
            )
        if self._te_mode == "torch":
            return super()._load_text_encoder_and_tokenizer(
                checkpoint_dir=checkpoint_dir, device=device
            )
        path = os.path.join(checkpoint_dir, "Qwen3-Embedding-0.6B")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Text encoder not found at {path}")
        if self._te_mode == "compare":
            super()._load_text_encoder_and_tokenizer(checkpoint_dir=checkpoint_dir, device=device)
            self._torch_text_encoder = self.text_encoder
        else:
            self.text_tokenizer = AutoTokenizer.from_pretrained(path)
        # Upstream checks ``self.text_encoder is not None``; anything that tries to call
        # it as a torch module fails loudly.
        self.text_encoder = MLXTextEncoder.load(path)
        logger.info("[lean] text encoder on MLX (bf16 weights, fp32 compute)")
        return path

    def _to_torch(self, arr: np.ndarray) -> torch.Tensor:
        """Numpy float32 to a torch tensor on the handler device and dtype."""
        return torch.from_numpy(arr).to(self.device).to(self.dtype)

    def infer_text_embeddings(self, text_token_idss):
        """MLX replacement for upstream's ``text_encoder(...).last_hidden_state``."""
        if self._te_mode == "torch":
            return super().infer_text_embeddings(text_token_idss)
        out = self._to_torch(self.text_encoder.hidden_states(text_token_idss.cpu().numpy()))
        if self._te_mode == "compare":
            with torch.inference_mode():
                ref = self._torch_text_encoder(input_ids=text_token_idss).last_hidden_state
            diff_report("TE", "hidden_states", out, ref)
        return out

    def infer_lyric_embeddings(self, lyric_token_ids):
        """MLX replacement for upstream's ``text_encoder.embed_tokens(...)``."""
        if self._te_mode == "torch":
            return super().infer_lyric_embeddings(lyric_token_ids)
        out = self._to_torch(self.text_encoder.embed(lyric_token_ids.cpu().numpy()))
        if self._te_mode == "compare":
            with torch.inference_mode():
                ref = self._torch_text_encoder.embed_tokens(lyric_token_ids)
            diff_report("TE", "lyric_embeddings", out, ref)
        return out
