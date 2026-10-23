"""Run the frozen-input replay once and save what the MLX condition bridges receive.

Usage on the Mac mini, from ~/acestep-f0 (same environment as run_f0.sh):
    F0_ENGINE=lean F0_REPLAY=$PWD/fixed.json F0_TAG=n3_capture \
        ACE-Step-1.5/.venv/bin/python ~/acestep-mlx/bench/n3/capture_inputs.py
Writes the first call's inputs of the encoder and detokenizer bridges to
``$N3_CAPTURE`` (default ~/acestep-f0/n3_inputs.npz) for noise_floor.py.
"""

import inspect
import os
import runpy
import sys

import numpy as np

HOME = os.path.expanduser("~")
sys.path[:0] = [f"{HOME}/acestep-mlx/engine", f"{HOME}/acestep-f0/ACE-Step-1.5"]

from acestep_mlx import condition_bridge as cb  # noqa: E402

OUT = os.environ.get("N3_CAPTURE", f"{HOME}/acestep-f0/n3_inputs.npz")
captured = {}


def capture_first_call(cls, prefix: str) -> None:
    """Wrap ``cls.forward`` so its first call's tensor arguments land in ``captured``."""
    original = cls.forward
    signature = inspect.signature(original)

    def forward(self, *args, **kwargs):
        if not any(key.startswith(prefix + ".") for key in captured):
            bound = signature.bind(self, *args, **kwargs)
            for name, value in bound.arguments.items():
                if name != "self":
                    captured[f"{prefix}.{name}"] = value.detach().cpu().numpy()
        return original(self, *args, **kwargs)

    cls.forward = forward


capture_first_call(cb.EncoderBridge, "encoder")
capture_first_call(cb.DetokenizerBridge, "detokenizer")
code = None
try:
    runpy.run_path(f"{HOME}/acestep-f0/f0_measure.py", run_name="__main__")
except SystemExit as stop:
    code = stop.code
np.savez(OUT, **captured)
print(f"saved {sorted(captured)} to {OUT}; f0_measure exit {code}")
