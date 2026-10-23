"""Difference report used by the compare modes (MLX output vs the torch module it replaces)."""

import torch
from loguru import logger


def diff_report(tag: str, name: str, ours: torch.Tensor, ref: torch.Tensor) -> None:
    """Log max/mean absolute difference and the worst per-token cosine similarity."""
    a, b = ours.float().flatten(0, -2), ref.float().flatten(0, -2)
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1).min().item()
    logger.info(f"[lean] {tag} compare {name}: max abs {(a - b).abs().max().item():.3e}, "
                f"mean abs {(a - b).abs().mean().item():.3e}, min token cosine {cos:.8f}, "
                f"bit-identical {torch.equal(a, b)}")
