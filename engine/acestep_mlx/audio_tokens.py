"""MLX ports of upstream ``AceStepAudioTokenizer`` and ``AudioTokenDetokenizer``.

The tokenizer pools each window of ``pool_window_size`` 25 Hz latent frames into one
5 Hz vector and quantizes it with residual FSQ; the detokenizer expands each 5 Hz
vector back into a window of frames. Both run their windows as independent sequences.
"""

from typing import Tuple

import mlx.core as mx
import mlx.nn as nn

from acestep_mlx.cond_layers import MLXEncoderBody
from acestep_mlx.fsq import MLXResidualFSQ


class MLXAttentionPooler(MLXEncoderBody):
    """Upstream ``AttentionPooler``: a learned token prepended to each window, its output kept."""

    def __init__(self, config):
        """Build from an encoder config."""
        super().__init__(config, config.num_attention_pooler_hidden_layers)
        self.embed_tokens = nn.Linear(config.hidden_size, config.hidden_size)
        self.special_token = mx.zeros((1, 1, config.hidden_size))

    def __call__(self, x: mx.array) -> mx.array:
        """``[B, T, P, D]`` windows to ``[B, T, D]`` pooled vectors."""
        b, t, p, d = x.shape
        x = self.embed_tokens(x)
        special = mx.broadcast_to(self.special_token[:, :, None, :], (b, t, 1, d))
        x = mx.concatenate([special.astype(x.dtype), x], axis=2).reshape(b * t, p + 1, d)
        return self.encode(x)[:, 0, :].reshape(b, t, d)


class MLXDetokenizer(MLXEncoderBody):
    """Upstream ``AudioTokenDetokenizer``: each 5 Hz vector plus learned per-frame offsets."""

    def __init__(self, config):
        """Build from an encoder config."""
        super().__init__(config, config.num_attention_pooler_hidden_layers)
        self.pool_window_size = config.pool_window_size
        self.embed_tokens = nn.Linear(config.hidden_size, config.hidden_size)
        self.special_tokens = mx.zeros((1, config.pool_window_size, config.hidden_size))
        self.proj_out = nn.Linear(config.hidden_size, config.audio_acoustic_hidden_dim)

    def __call__(self, x: mx.array) -> mx.array:
        """``[B, T, D]`` quantized vectors to ``[B, T * P, audio_acoustic_hidden_dim]``."""
        b, t, d = x.shape
        p = self.pool_window_size
        x = self.embed_tokens(x)[:, :, None, :] + self.special_tokens[None]
        h = self.proj_out(self.encode(x.reshape(b * t, p, d)))
        return h.reshape(b, t * p, -1)


class MLXAudioTokenizer(nn.Module):
    """Upstream ``AceStepAudioTokenizer``: project, pool windows, quantize."""

    def __init__(self, config):
        """Build from an encoder config."""
        super().__init__()
        self.audio_acoustic_proj = nn.Linear(config.audio_acoustic_hidden_dim, config.hidden_size)
        self.attention_pooler = MLXAttentionPooler(config)
        self.quantizer = MLXResidualFSQ(
            config.fsq_dim, config.fsq_input_levels, config.fsq_input_num_quantizers
        )

    def __call__(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        """Windows ``[B, T, P, acoustic_dim]`` to quantized ``[B, T, fsq_dim]`` and indices."""
        return self.quantizer(self.attention_pooler(self.audio_acoustic_proj(x)))
