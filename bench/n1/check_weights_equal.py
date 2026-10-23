"""N1 oracle check: MLX decoder weights read from safetensors equal upstream's conversion.

Upstream ``convert_decoder_weights`` (fed one torch tensor at a time, to keep memory
low) is the oracle; ``acestep_mlx.dit_weights.load_mlx_decoder`` is the candidate.
Every parameter must match bit for bit after both are viewed as float32. Exit 0 only
if all keys match and the key sets are identical.
"""

import os
import sys

import numpy as np

PROJECT = os.path.expanduser("~/acestep-f0/ACE-Step-1.5")
ENGINE = os.path.expanduser("~/acestep-mlx/engine")
CKPT_DIR = os.path.join(PROJECT, "checkpoints", "acestep-v15-turbo")
sys.path[:0] = [PROJECT, ENGINE]


class _OneTensorModel:
    """Minimal stand-in exposing ``decoder.state_dict()`` with a single entry."""

    def __init__(self, key, tensor):
        self.decoder = self
        self._sd = {key: tensor}

    def state_dict(self):
        """Return the single-entry state dict."""
        return self._sd


def main() -> int:
    """Compare oracle and candidate parameter by parameter."""
    import mlx.core as mx
    import torch
    from mlx.utils import tree_flatten
    from transformers import AutoModel

    from acestep.models.mlx.dit_convert import convert_decoder_weights
    from acestep_mlx.dit_weights import load_mlx_decoder

    model = AutoModel.from_pretrained(CKPT_DIR, trust_remote_code=True, dtype=torch.bfloat16)
    candidate = dict(tree_flatten(load_mlx_decoder(os.path.join(CKPT_DIR, "model.safetensors"),
                                                   model.config).parameters()))
    oracle_keys, mismatches, checked = set(), [], 0
    for key, tensor in model.decoder.state_dict().items():
        for new_key, arr in convert_decoder_weights(_OneTensorModel(key, tensor.float())):
            oracle_keys.add(new_key)
            ours = candidate.get(new_key)
            if ours is None:
                mismatches.append((new_key, "missing in candidate"))
                continue
            a = np.array(arr.astype(mx.float32))
            b = np.array(ours.astype(mx.float32))
            checked += 1
            if a.shape != b.shape or not np.array_equal(a, b):
                mismatches.append((new_key, f"shape {a.shape} vs {b.shape}, "
                                   f"max abs diff {np.abs(a - b).max() if a.shape == b.shape else 'n/a'}"))
    extra = sorted(set(candidate) - oracle_keys)
    dtypes = sorted({str(v.dtype) for v in candidate.values()})
    print(f"checked {checked} params, mismatches {len(mismatches)}, extra keys {len(extra)}, "
          f"candidate dtypes {dtypes}")
    for m in mismatches[:10]:
        print("MISMATCH", m)
    for k in extra[:10]:
        print("EXTRA", k)
    return 0 if not mismatches and not extra and checked > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
