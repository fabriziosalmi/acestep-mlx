"""Load the MLX DiT decoder straight from the checkpoint, without a torch copy.

Upstream (``acestep/models/mlx/dit_convert.py``) builds the MLX decoder from an
already-loaded torch module, which forces the whole decoder to exist in torch
first. The checkpoint is stored in bf16, so reading it with ``mx.load`` keeps the
exact values while skipping torch entirely. The key remapping mirrors upstream.
"""

from typing import Callable, Iterable, List, Optional, Tuple

DECODER_PREFIX = "decoder."


def remap_decoder_key(key: str) -> Tuple[Optional[str], Optional[Callable]]:
    """Map a torch decoder parameter name to its MLX name and layout transform.

    Args:
        key: Parameter name relative to the decoder (no ``decoder.`` prefix).

    Returns:
        ``(new_key, transform)``. ``new_key`` is None for entries MLX recomputes
        (rotary embedding buffers). ``transform`` is None when the tensor is used
        as-is, otherwise a function applied to the array.
    """
    if "rotary_emb" in key:
        return None, None
    if key.startswith("proj_in.1."):
        new_key = key.replace("proj_in.1.", "proj_in.", 1)
        # Conv1d weight: torch [out, in, K] -> MLX [out, K, in]
        transform = (lambda a: a.swapaxes(1, 2)) if new_key.endswith(".weight") else None
        return new_key, transform
    if key.startswith("proj_out.1."):
        new_key = key.replace("proj_out.1.", "proj_out.", 1)
        # ConvTranspose1d weight: torch [in, out, K] -> MLX [out, K, in]
        transform = (lambda a: a.transpose(1, 2, 0)) if new_key.endswith(".weight") else None
        return new_key, transform
    return key, None


def select_decoder_weights(items: Iterable[Tuple[str, object]]) -> List[Tuple[str, object]]:
    """Keep decoder entries of a full-model state dict and remap them for MLX.

    Args:
        items: ``(name, array)`` pairs for the whole model.

    Returns:
        ``(mlx_name, array)`` pairs for the decoder only, in input order.
    """
    out = []
    for name, arr in items:
        if not name.startswith(DECODER_PREFIX):
            continue
        new_key, transform = remap_decoder_key(name[len(DECODER_PREFIX):])
        if new_key is None:
            continue
        out.append((new_key, transform(arr) if transform else arr))
    return out


def load_mlx_decoder(checkpoint_file: str, config) -> "object":
    """Build an evaluated ``MLXDiTDecoder`` from a model safetensors file.

    Args:
        checkpoint_file: Path to the DiT ``model.safetensors``.
        config: The model config used by upstream ``MLXDiTDecoder.from_config``.

    Returns:
        The MLX decoder with weights loaded in their stored dtype.

    Raises:
        ValueError: If the file holds keys the decoder does not expect, or misses some
            (``load_weights`` runs in strict mode).
    """
    import mlx.core as mx
    from acestep.models.mlx.dit_model import MLXDiTDecoder

    arrays = mx.load(checkpoint_file)
    weights = select_decoder_weights(arrays.items())
    del arrays
    decoder = MLXDiTDecoder.from_config(config)
    decoder.load_weights(weights, strict=True)
    mx.eval(decoder.parameters())
    decoder.materialize_static_buffers()
    return decoder
