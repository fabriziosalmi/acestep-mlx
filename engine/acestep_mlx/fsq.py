"""MLX port of ``vector_quantize_pytorch.ResidualFSQ`` as upstream configures it.

Upstream builds ``ResidualFSQ(dim, levels, num_quantizers)`` with the library defaults:
symmetry-preserving FSQ layers, hard clamp, and a soft clamp of the input to
``1 + 1 / (levels - 1)``. Checked against vector-quantize-pytorch 1.27.20. Each
elementwise step keeps the library's operation order, so codes and indices match
bit for bit; only the two projections are matmuls.
"""

from typing import Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class MLXResidualFSQ(nn.Module):
    """Residual finite scalar quantizer with input and output projections."""

    def __init__(self, dim: int, levels: Sequence[int], num_quantizers: int):
        """Create the projections; ``levels`` and their mixed-radix basis are constants."""
        super().__init__()
        self.project_in = nn.Linear(dim, len(levels))
        self.project_out = nn.Linear(len(levels), dim)
        self._levels = mx.array(list(levels), dtype=mx.int32)
        self._basis = mx.array(np.cumprod([1] + list(levels[:-1])).astype(np.int32))
        self._num_quantizers = num_quantizers
        self._scales = [mx.array(list(levels), dtype=mx.float32) ** -q
                        for q in range(num_quantizers)]

    def _level_codes(self, indices: mx.array) -> mx.array:
        """FSQ ``_indices_to_codes`` with symmetry: integer index to codes in [-1, 1]."""
        level_idx = (indices[..., None] // self._basis) % self._levels
        return level_idx * (2.0 / (self._levels - 1)) - 1.0

    def _bound(self, z: mx.array) -> mx.array:
        """FSQ ``symmetry_preserving_bound`` with ``hard_clamp=True``."""
        levels_minus_1 = self._levels - 1
        scale = 2.0 / levels_minus_1
        bracket = (levels_minus_1 * (mx.clip(z, -1.0, 1.0) + 1) / 2.0) + 0.5
        return scale * mx.floor(bracket) - 1.0

    def _codes_to_indices(self, codes: mx.array) -> mx.array:
        """FSQ ``codes_to_indices`` with symmetry."""
        zhat = (codes + 1.0) / (2.0 / (self._levels - 1))
        return mx.round((zhat * self._basis).sum(axis=-1)).astype(mx.int32)

    def codes_from_indices(self, indices: mx.array) -> mx.array:
        """Summed, scaled codes ``[..., len(levels)]`` for indices ``[..., num_quantizers]``.

        Raises:
            ValueError: If the last axis is not ``num_quantizers`` (upstream would
                need quantizer dropout for fewer).
        """
        if indices.shape[-1] != self._num_quantizers:
            raise ValueError(
                f"expected {self._num_quantizers} quantizer indices, got {indices.shape[-1]}"
            )
        total = None
        for q in range(self._num_quantizers):
            codes = self._level_codes(indices[..., q]) * self._scales[q]
            total = codes if total is None else total + codes
        return total

    def output_from_indices(self, indices: mx.array) -> mx.array:
        """Upstream ``get_output_from_indices``: indices to projected embeddings ``[..., dim]``."""
        return self.project_out(self.codes_from_indices(indices))

    def __call__(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        """Quantize ``[..., dim]``; returns projected codes and int32 indices ``[..., Q]``."""
        x = self.project_in(x)
        clamp = 1 + 1 / (self._levels.astype(mx.float32) - 1)
        residual = mx.tanh(x / clamp) * clamp
        quantized_out, all_indices = 0.0, []
        for scale in self._scales:
            codes = self._bound(residual / scale)
            all_indices.append(self._codes_to_indices(codes))
            quantized = codes * scale
            residual = residual - quantized
            quantized_out = quantized_out + quantized
        return self.project_out(quantized_out), mx.stack(all_indices, axis=-1)
