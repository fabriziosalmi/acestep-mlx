"""Compare MLX, torch MPS fp32 and torch CPU fp32 on the captured real inputs.

The MPS run is what upstream executes on a Mac; CPU vs MPS is the disagreement between
two torch backends computing the same fp32 graph, i.e. the noise floor for MLX vs MPS.
Usage on the Mac mini: ACE-Step-1.5/.venv/bin/python ~/acestep-mlx/bench/n3/noise_floor.py
"""

import copy
import os
import sys

import numpy as np

HOME = os.path.expanduser("~")
PROJECT = f"{HOME}/acestep-f0/ACE-Step-1.5"
CKPT = f"{PROJECT}/checkpoints/acestep-v15-turbo"
sys.path[:0] = [f"{HOME}/acestep-mlx/engine", PROJECT]

import mlx.core as mx  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from acestep.models.common.configuration_acestep_v15 import AceStepConfig  # noqa: E402
from acestep.models.turbo import modeling_acestep_v15_turbo as up  # noqa: E402

from acestep_mlx.condition_encoder import encoder_config  # noqa: E402
from acestep_mlx.condition_weights import load_condition_modules, select_prefix  # noqa: E402


def stats(a: np.ndarray, b: np.ndarray) -> str:
    """Max/mean abs difference and min per-token cosine of two ``[..., D]`` arrays."""
    a = a.reshape(-1, a.shape[-1]).astype(np.float64)
    b = b.reshape(-1, b.shape[-1]).astype(np.float64)
    cos = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    return (f"max abs {np.abs(a - b).max():.3e}  mean abs {np.abs(a - b).mean():.3e}  "
            f"min cos {cos.min():.8f}  identical {np.array_equal(a, b)}")


def torch_module(cls, cfg, state, prefix):
    """Upstream module in fp32 with the checkpoint's bf16 weights, eval mode, sdpa."""
    module = cls(cfg)
    weights = {k: v.float() for k, v in select_prefix(state.items(), prefix)}
    module.load_state_dict(weights, strict=True)
    return module.eval()


def main() -> None:
    """Run the three backends on the captured inputs and print pairwise differences."""
    inputs = dict(np.load(os.environ.get("N3_CAPTURE", f"{HOME}/acestep-f0/n3_inputs.npz")))
    config = AceStepConfig.from_json_file(f"{CKPT}/config.json")
    config._attn_implementation = "sdpa"
    cfg = encoder_config(config)
    state = load_file(f"{CKPT}/model.safetensors")
    mlx_mods = load_condition_modules(f"{CKPT}/model.safetensors", config)
    enc_args = {k.split(".", 1)[1]: v for k, v in inputs.items() if k.startswith("encoder.")}
    det_x = inputs["detokenizer.x"]
    for name, cls, prefix in (("encoder", up.AceStepConditionEncoder, "encoder"),
                              ("detokenizer", up.AudioTokenDetokenizer, "detokenizer")):
        cpu = torch_module(cls, cfg, state, prefix)
        mps = copy.deepcopy(cpu).to("mps")
        outs = {}
        with torch.inference_mode():
            for dev, mod in (("cpu", cpu), ("mps", mps)):
                if name == "encoder":
                    args = {k: torch.from_numpy(v).to(dev) for k, v in enc_args.items()}
                    outs[dev] = mod(**args)[0].float().cpu().numpy()
                else:
                    outs[dev] = mod(torch.from_numpy(det_x).to(dev)).float().cpu().numpy()
        if name == "encoder":
            h, _ = mlx_mods["encoder"](
                mx.array(enc_args["text_hidden_states"]), enc_args["text_attention_mask"],
                mx.array(enc_args["lyric_hidden_states"]), enc_args["lyric_attention_mask"],
                mx.array(enc_args["refer_audio_acoustic_hidden_states_packed"]),
                enc_args["refer_audio_order_mask"],
            )
        else:
            h = mlx_mods["detokenizer"](mx.array(det_x))
        outs["mlx"] = np.array(h)
        print(f"{name} shape {outs['mlx'].shape}")
        for x, y in (("mlx", "mps"), ("cpu", "mps"), ("mlx", "cpu")):
            print(f"  {x:3s} vs {y:3s}: {stats(outs[x], outs[y])}")
        del cpu, mps
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
