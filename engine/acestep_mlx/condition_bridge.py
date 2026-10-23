"""Weightless torch modules that run the MLX condition modules in place of upstream's.

Upstream code calls ``model.encoder(...)``, ``model.tokenizer(...)``,
``model.tokenizer.quantizer.get_output_from_indices(...)`` and ``model.detokenizer(...)``
with torch tensors. These bridges keep those call signatures, convert at the boundary
and hold no torch parameters. With a ``twin`` (the original torch module) they also run
it on the same inputs and log the difference, while still returning the MLX result.
"""

from typing import Optional

import mlx.core as mx
import numpy as np
import torch
from loguru import logger

from acestep_mlx.diffs import diff_report


def to_mx(t: torch.Tensor) -> mx.array:
    """Torch tensor to an MLX array (floating point as fp32)."""
    t = t.detach()
    if t.is_floating_point():
        t = t.float()
    return mx.array(t.cpu().numpy())


def to_np(t: torch.Tensor) -> np.ndarray:
    """Torch mask or index tensor to numpy."""
    return t.detach().cpu().numpy()


def to_torch(a, like: torch.Tensor) -> torch.Tensor:
    """MLX array or numpy array to torch on ``like``'s device; floats take ``like``'s dtype."""
    out = torch.from_numpy(np.array(a)).to(like.device)
    return out.to(like.dtype) if out.is_floating_point() else out


class _Bridge(torch.nn.Module):
    """Shared plumbing: the MLX module, an optional torch twin, compare logging."""

    def __init__(self, name: str, mlx_module, twin: Optional[torch.nn.Module]):
        """``twin`` is registered as a submodule so it follows ``.to(device)``."""
        super().__init__()
        self._name = name
        self._mlx = mlx_module
        self.twin = twin

    def _compare(self, label: str, ours: torch.Tensor, ref: torch.Tensor) -> None:
        """Log float differences, or the share of equal entries for integer outputs."""
        if ref.is_floating_point():
            diff_report("COND", f"{self._name}.{label}", ours, ref)
            return
        equal = (ours.to(ref.dtype) == ref).float().mean().item()
        logger.info(f"[lean] COND compare {self._name}.{label}: equal {equal:.6f} of "
                    f"{ref.numel()} entries")


class EncoderBridge(_Bridge):
    """Stands in for ``model.encoder`` (``AceStepConditionEncoder``)."""

    def forward(self, text_hidden_states, text_attention_mask, lyric_hidden_states,
                lyric_attention_mask, refer_audio_acoustic_hidden_states_packed,
                refer_audio_order_mask):
        """Same arguments and outputs as upstream: packed states and their mask."""
        h, mask = self._mlx(
            to_mx(text_hidden_states), to_np(text_attention_mask),
            to_mx(lyric_hidden_states), to_np(lyric_attention_mask),
            to_mx(refer_audio_acoustic_hidden_states_packed), to_np(refer_audio_order_mask),
        )
        h_t = to_torch(h, text_hidden_states)
        mask_t = torch.from_numpy(mask).to(text_hidden_states.device)
        if self.twin is not None:
            ref_h, ref_mask = self.twin(
                text_hidden_states=text_hidden_states, text_attention_mask=text_attention_mask,
                lyric_hidden_states=lyric_hidden_states,
                lyric_attention_mask=lyric_attention_mask,
                refer_audio_acoustic_hidden_states_packed=refer_audio_acoustic_hidden_states_packed,
                refer_audio_order_mask=refer_audio_order_mask,
            )
            self._compare("hidden_states", h_t, ref_h)
            self._compare("mask", mask_t, ref_mask)
        return h_t, mask_t


class DetokenizerBridge(_Bridge):
    """Stands in for ``model.detokenizer``."""

    def forward(self, x):
        """``[B, T, D]`` quantized to ``[B, T * P, acoustic_dim]``."""
        out = to_torch(self._mlx(to_mx(x)), x)
        if self.twin is not None:
            self._compare("out", out, self.twin(x))
        return out


class QuantizerBridge(_Bridge):
    """Stands in for ``model.tokenizer.quantizer``; only index decoding is used upstream."""

    def get_output_from_indices(self, indices):
        """Indices ``[B, N, Q]`` to projected embeddings ``[B, N, fsq_dim]`` (fp32)."""
        out = torch.from_numpy(np.array(self._mlx.output_from_indices(mx.array(to_np(indices)))))
        out = out.to(indices.device)
        if self.twin is not None:
            ref = self.twin.get_output_from_indices(indices)
            ours_codes = np.array(self._mlx.codes_from_indices(mx.array(to_np(indices))))
            ref_codes = self.twin.get_codes_from_indices(indices).sum(0).cpu().numpy()
            logger.info(f"[lean] COND compare {self._name}.codes bit-identical "
                        f"{np.array_equal(ours_codes, ref_codes)}")
            self._compare("output", out, ref)
        return out


class TokenizerBridge(_Bridge):
    """Stands in for ``model.tokenizer`` (``AceStepAudioTokenizer``)."""

    def __init__(self, name: str, mlx_module, twin: Optional[torch.nn.Module]):
        """Expose the quantizer bridge as ``.quantizer``, as upstream code expects."""
        super().__init__(name, mlx_module, twin)
        self.quantizer = QuantizerBridge(
            f"{name}.quantizer", mlx_module.quantizer, None if twin is None else twin.quantizer
        )

    def forward(self, hidden_states):
        """Windows ``[B, T, P, acoustic_dim]`` to quantized ``[B, T, D]`` and int32 indices."""
        q, idx = self._mlx(to_mx(hidden_states))
        q_t = to_torch(q, hidden_states)
        idx_t = torch.from_numpy(np.array(idx)).to(hidden_states.device)
        if self.twin is not None:
            ref_q, ref_idx = self.twin(hidden_states)
            self._compare("quantized", q_t, ref_q)
            self._compare("indices", idx_t, ref_idx)
        return q_t, idx_t
