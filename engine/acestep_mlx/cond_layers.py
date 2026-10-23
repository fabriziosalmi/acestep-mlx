"""MLX encoder layers shared by the condition encoder, audio tokenizer and detokenizer.

Upstream (``modeling_acestep_v15_turbo.py``) builds all four from the same Qwen3 block:
pre-norm bidirectional self-attention with QK RMSNorm and RoPE, then a SwiGLU MLP.
Sliding layers see keys within ``sliding_window`` positions on either side, and the
padding mask hides keys only. Parameter names match the checkpoint, so bf16 weights
load as stored; fp32 activations make every op compute in fp32.
"""

from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from acestep.models.mlx.dit_model import MLXAttention, MLXSwiGLUMLP

# Upstream's additive mask uses the dtype minimum, not -inf: a row with every key
# masked then averages all values instead of producing NaN.
MASK_FILL = float(np.finfo(np.float32).min)

CosSin = Tuple[mx.array, mx.array]


def rope_cos_sin(seq_len: int, head_dim: int, base: float) -> CosSin:
    """Rotary tables for positions ``0..seq_len-1``, each ``[1, 1, L, head_dim]`` in fp32."""
    inv_freq = 1.0 / (base ** (mx.arange(0, head_dim, 2).astype(mx.float32) / head_dim))
    freqs = mx.arange(seq_len).astype(mx.float32)[:, None] * inv_freq[None, :]
    emb = mx.concatenate([freqs, freqs], axis=-1)
    return mx.cos(emb)[None, None], mx.sin(emb)[None, None]


def _additive(valid: mx.array) -> mx.array:
    """Boolean keep-mask to upstream's additive fp32 mask."""
    return mx.where(valid, mx.array(0.0, mx.float32), mx.array(MASK_FILL, mx.float32))


def encoder_masks(
    seq_len: int, key_mask: Optional[mx.array], sliding_window: Optional[int]
) -> Dict[str, Optional[mx.array]]:
    """Additive masks per layer type, as upstream ``create_4d_mask`` builds them (bidirectional).

    Args:
        seq_len: Sequence length L.
        key_mask: ``[B, L]`` with 1 for valid keys, or None when every key is valid.
        sliding_window: Window of sliding layers; None when the config disables it, in
            which case upstream gives sliding layers no mask at all.

    Returns:
        ``{"full_attention": mask, "sliding_attention": mask}``, each ``[B|1, 1, L, L]`` or
        None where no position would be hidden.
    """
    keys = None if key_mask is None else key_mask.astype(mx.bool_)[:, None, None, :]
    full = None if keys is None else _additive(keys)
    sliding = None
    if sliding_window is not None:
        idx = mx.arange(seq_len)
        near = (mx.abs(idx[:, None] - idx[None, :]) <= sliding_window)[None, None]
        if keys is not None:
            sliding = _additive(near & keys)
        elif seq_len > sliding_window + 1:
            sliding = _additive(near)
    return {"full_attention": full, "sliding_attention": sliding}


class MLXEncoderLayer(nn.Module):
    """Upstream ``AceStepEncoderLayer``: residual self-attention, then residual SwiGLU MLP."""

    def __init__(self, config, layer_idx: int):
        """Build layer ``layer_idx`` of an encoder described by an ``AceStepConfig``."""
        super().__init__()
        self.attention_type = config.layer_types[layer_idx]
        head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        self.self_attn = MLXAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=head_dim,
            rms_norm_eps=config.rms_norm_eps,
            attention_bias=config.attention_bias,
            layer_idx=layer_idx,
        )
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = MLXSwiGLUMLP(config.hidden_size, config.intermediate_size)

    def __call__(self, h: mx.array, cos_sin: CosSin, mask: Optional[mx.array]) -> mx.array:
        """``[B, L, D]`` to ``[B, L, D]``."""
        h = h + self.self_attn(
            self.input_layernorm(h), position_cos_sin=cos_sin, attention_mask=mask
        )
        return h + self.mlp(self.post_attention_layernorm(h))


class MLXEncoderBody(nn.Module):
    """Shared body of upstream's encoders: ``layers`` then ``norm``, names as in the checkpoint."""

    def __init__(self, config, num_layers: int, active_layers: Optional[int] = None):
        """Create ``num_layers`` encoder layers and the final RMSNorm.

        ``active_layers`` caps how many run, as upstream's lyric and timbre encoders slice
        ``layers[:num_hidden_layers]``; None runs them all.
        """
        super().__init__()
        self._active_layers = num_layers if active_layers is None else active_layers
        self.layers: List[MLXEncoderLayer] = [
            MLXEncoderLayer(config, idx) for idx in range(num_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self._head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        self._rope_base = float(config.rope_theta)
        self._sliding_window = config.sliding_window if config.use_sliding_window else None

    def encode(self, h: mx.array, key_mask: Optional[mx.array] = None) -> mx.array:
        """Run every layer and the final norm over ``[B, L, D]`` hidden states."""
        seq_len = h.shape[1]
        cos_sin = rope_cos_sin(seq_len, self._head_dim, self._rope_base)
        masks = encoder_masks(seq_len, key_mask, self._sliding_window)
        for layer in self.layers[: self._active_layers]:
            h = layer(h, cos_sin, masks[layer.attention_type])
        return self.norm(h)
