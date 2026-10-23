"""Handler mixin: the DiT's condition encoder, audio tokenizer and detokenizer run on MLX.

``ACESTEP_LEAN_CONDITION`` selects ``mlx`` (default: the torch modules are replaced by
weightless bridges before the model moves to the device), ``torch`` (upstream modules),
or ``compare`` (MLX outputs, with the torch modules kept to log the difference).
"""

import gc
import os

from loguru import logger

from acestep_mlx.condition_bridge import DetokenizerBridge, EncoderBridge, TokenizerBridge
from acestep_mlx.condition_weights import load_condition_modules

MODES = ("mlx", "torch", "compare")
BRIDGES = (("encoder", EncoderBridge), ("tokenizer", TokenizerBridge),
           ("detokenizer", DetokenizerBridge))


def condition_mode() -> str:
    """The selected mode.

    Raises:
        ValueError: For a value outside ``mlx``, ``torch``, ``compare``.
    """
    mode = os.environ.get("ACESTEP_LEAN_CONDITION", "mlx")
    if mode not in MODES:
        raise ValueError(f"ACESTEP_LEAN_CONDITION must be one of {MODES}, not {mode}")
    return mode


class MLXConditionMixin:
    """Swaps ``self.model``'s condition-side submodules for MLX bridges."""

    def _install_mlx_condition(self, checkpoint_file: str) -> None:
        """Load the MLX modules from ``checkpoint_file`` and install their bridges.

        Call it while the torch model is still on CPU, so in ``mlx`` mode the replaced
        modules never reach the device.
        """
        mode = condition_mode()
        if mode == "torch":
            return
        modules = load_condition_modules(checkpoint_file, self.config)
        for name, bridge in BRIDGES:
            twin = getattr(self.model, name) if mode == "compare" else None
            setattr(self.model, name, bridge(name, modules[name], twin))
        gc.collect()
        logger.info(f"[lean] condition encoder, tokenizer and detokenizer on MLX ({mode} mode)")
