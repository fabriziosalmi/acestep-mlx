"""MLX port of upstream ``AceStepConditionEncoder``: text projector, lyric and timbre encoders.

Integer bookkeeping (timbre unpacking, sequence packing) runs in numpy on the masks and
reproduces upstream's ordering exactly; only gathers touch the hidden states, so those
steps add no numerical difference.
"""

import copy
from typing import Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from acestep_mlx.cond_layers import MLXEncoderBody


def encoder_config(config):
    """Config of the encoder-side modules, as upstream derives it for 4B models.

    Returns ``config`` itself when ``encoder_hidden_size`` is unset or equal to
    ``hidden_size`` (all 2B models).
    """
    enc_hs = getattr(config, "encoder_hidden_size", None) or config.hidden_size
    if enc_hs == config.hidden_size:
        return config
    out = copy.deepcopy(config)
    out.hidden_size = enc_hs
    for field in ("intermediate_size", "num_attention_heads", "num_key_value_heads"):
        setattr(out, field, getattr(config, f"encoder_{field}", None) or getattr(config, field))
    return out


def gather_rows(h: mx.array, idx: np.ndarray) -> mx.array:
    """``h[b, idx[b, j]]`` for ``h`` ``[B, L, D]`` and integer ``idx`` ``[B, L']``."""
    index = mx.array(idx.astype(np.int32))[..., None]
    return mx.take_along_axis(h, mx.broadcast_to(index, (*idx.shape, h.shape[-1])), axis=1)


def pack_sequences(
    h1: mx.array, h2: mx.array, mask1: np.ndarray, mask2: np.ndarray
) -> Tuple[mx.array, np.ndarray]:
    """Concatenate two sequences and move valid tokens first, keeping their order.

    Mirrors upstream ``pack_sequences`` (stable descending argsort on the mask).

    Returns:
        Packed hidden states ``[B, L1+L2, D]`` and the boolean mask of valid positions.
    """
    mask_cat = np.concatenate([mask1, mask2], axis=1)
    order = np.argsort(-mask_cat.astype(np.int64), axis=1, kind="stable")
    packed = gather_rows(mx.concatenate([h1, h2], axis=1), order)
    lengths = mask_cat.astype(np.int64).sum(axis=1)
    return packed, np.arange(mask_cat.shape[1])[None, :] < lengths[:, None]


def unpack_timbre(emb: mx.array, order_mask: np.ndarray) -> Tuple[mx.array, np.ndarray]:
    """Spread packed per-reference embeddings ``[N, D]`` into ``[B, max_count, D]``.

    ``order_mask[i]`` is the batch item of reference ``i``; references keep their
    original order inside each item, empty slots are zeros, as in upstream.

    Returns:
        Unpacked embeddings and the int64 mask of filled slots ``[B, max_count]``.
    """
    order = order_mask.astype(np.int64)
    batch = int(order.max()) + 1
    max_count = int(np.bincount(order, minlength=batch).max())
    slot_source = np.full(batch * max_count, -1, dtype=np.int64)
    seen = np.zeros(batch, dtype=np.int64)
    for i, b in enumerate(order):
        slot_source[b * max_count + seen[b]] = i
        seen[b] += 1
    filled = slot_source >= 0
    rows = emb[mx.array(np.where(filled, slot_source, 0).astype(np.int32))]
    rows = mx.where(mx.array(filled)[:, None], rows, mx.zeros_like(rows))
    return rows.reshape(batch, max_count, -1), filled.astype(np.int64).reshape(batch, max_count)


class MLXLyricEncoder(MLXEncoderBody):
    """Upstream ``AceStepLyricEncoder``: project lyric token embeddings, then encode."""

    def __init__(self, config):
        """Build from an encoder config."""
        super().__init__(
            config, config.num_lyric_encoder_hidden_layers, config.num_hidden_layers
        )
        self.embed_tokens = nn.Linear(config.text_hidden_dim, config.hidden_size)

    def __call__(self, embeds: mx.array, mask: np.ndarray) -> mx.array:
        """``[B, L, text_hidden_dim]`` and key mask ``[B, L]`` to ``[B, L, D]``."""
        return self.encode(self.embed_tokens(embeds), mx.array(mask))


class MLXTimbreEncoder(MLXEncoderBody):
    """Upstream ``AceStepTimbreEncoder``: first-frame output of each reference latent."""

    def __init__(self, config):
        """Build from an encoder config."""
        super().__init__(
            config, config.num_timbre_encoder_hidden_layers, config.num_hidden_layers
        )
        self.embed_tokens = nn.Linear(config.timbre_hidden_dim, config.hidden_size)
        # Stored in the checkpoint; upstream's forward no longer prepends it.
        self.special_token = mx.zeros((1, 1, config.hidden_size))

    def __call__(self, packed: mx.array, order_mask: np.ndarray) -> Tuple[mx.array, np.ndarray]:
        """References ``[N, T, timbre_hidden_dim]`` to unpacked ``[B, max_count, D]`` and mask."""
        first = self.encode(self.embed_tokens(packed))[:, 0, :]
        return unpack_timbre(first, order_mask)


class MLXConditionEncoder(nn.Module):
    """Upstream ``AceStepConditionEncoder``: lyrics, then timbre, then text, packed."""

    def __init__(self, config):
        """Build from an encoder config."""
        super().__init__()
        self.text_projector = nn.Linear(config.text_hidden_dim, config.hidden_size, bias=False)
        self.lyric_encoder = MLXLyricEncoder(config)
        self.timbre_encoder = MLXTimbreEncoder(config)

    def __call__(
        self,
        text_hidden_states: mx.array,
        text_attention_mask: np.ndarray,
        lyric_hidden_states: mx.array,
        lyric_attention_mask: np.ndarray,
        refer_packed: mx.array,
        refer_order_mask: np.ndarray,
    ) -> Tuple[mx.array, np.ndarray]:
        """Return the packed cross-attention sequence and its boolean mask."""
        text = self.text_projector(text_hidden_states)
        lyric = self.lyric_encoder(lyric_hidden_states, lyric_attention_mask)
        timbre, timbre_mask = self.timbre_encoder(refer_packed, refer_order_mask)
        h, mask = pack_sequences(lyric, timbre, lyric_attention_mask, timbre_mask)
        return pack_sequences(h, text, mask, text_attention_mask)
