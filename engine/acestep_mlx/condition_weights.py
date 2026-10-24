"""Build the MLX condition-side modules straight from the DiT checkpoint.

The checkpoint stores ``encoder.*``, ``tokenizer.*`` and ``detokenizer.*`` in bf16 under
the same names the MLX modules use, so each module loads its prefix as stored, strictly.
"""

from typing import Dict, Iterable, List, Tuple

from acestep_mlx.audio_tokens import MLXAudioTokenizer, MLXDetokenizer
from acestep_mlx.condition_encoder import MLXConditionEncoder, encoder_config

MODULES = ("encoder", "tokenizer", "detokenizer")


def select_prefix(items: Iterable[Tuple[str, object]], prefix: str) -> List[Tuple[str, object]]:
    """``(name, array)`` pairs under ``prefix.``, with the prefix removed, in input order."""
    head = prefix + "."
    return [(name[len(head):], arr) for name, arr in items if name.startswith(head)]


def build_condition_modules(config) -> Dict[str, object]:
    """Uninitialized MLX ``encoder``, ``tokenizer`` and ``detokenizer`` for this config."""
    cfg = encoder_config(config)
    return {
        "encoder": MLXConditionEncoder(cfg),
        "tokenizer": MLXAudioTokenizer(cfg),
        "detokenizer": MLXDetokenizer(cfg),
    }


def load_condition_modules(checkpoint_file: str, config) -> Dict[str, object]:
    """Evaluated MLX condition modules with weights read from ``model.safetensors``.

    Raises:
        ValueError: If a module's checkpoint keys do not match its parameters (strict load).
    """
    import mlx.core as mx

    arrays = mx.load(checkpoint_file)
    modules = build_condition_modules(config)
    for name, module in modules.items():
        module.load_weights(select_prefix(arrays.items(), name), strict=True)
        mx.eval(module.parameters())
    del arrays
    return modules
