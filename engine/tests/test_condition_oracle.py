"""MLX condition modules against upstream's torch modules on a tiny random config.

Needs upstream ``acestep`` and torch importable (the Mac mini venv). Weights go from the
torch state dict into MLX with a strict load, so a name mismatch fails here too.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mlx.core as mx  # noqa: E402
import torch  # noqa: E402

from acestep.models.common.configuration_acestep_v15 import AceStepConfig  # noqa: E402
from acestep.models.turbo import modeling_acestep_v15_turbo as up  # noqa: E402

from acestep_mlx.audio_tokens import MLXAudioTokenizer, MLXDetokenizer  # noqa: E402
from acestep_mlx.cond_layers import encoder_masks  # noqa: E402
from acestep_mlx.condition_encoder import MLXConditionEncoder  # noqa: E402

REL_TOL = 1e-5


def tiny_config(**overrides) -> AceStepConfig:
    """Small config with every structural feature on: GQA, sliding window, two layer types."""
    kwargs = dict(
        hidden_size=64, intermediate_size=128, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, num_hidden_layers=3, num_lyric_encoder_hidden_layers=3,
        num_timbre_encoder_hidden_layers=2, num_attention_pooler_hidden_layers=2,
        sliding_window=4, text_hidden_dim=32, timbre_hidden_dim=8, audio_acoustic_hidden_dim=8,
        fsq_dim=64, pool_window_size=5,
        layer_types=["sliding_attention", "full_attention", "sliding_attention"],
    )
    kwargs.update(overrides)
    return AceStepConfig(**kwargs)


def randomized(module: torch.nn.Module, seed: int) -> torch.nn.Module:
    """Fill every parameter (norm weights included) with random values; eval mode."""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(torch.randn(p.shape, generator=gen) * 0.3 + (1.0 if p.ndim == 1 else 0.0))
    return module.eval()


def mlx_twin(mlx_module, torch_module):
    """Load the torch state dict into the MLX module, strictly."""
    weights = [(k, mx.array(v.numpy())) for k, v in torch_module.state_dict().items()]
    mlx_module.load_weights(weights, strict=True)
    return mlx_module


def assert_close(test, ours, ref):
    """Max abs difference relative to the reference's max magnitude is below REL_TOL."""
    ours, ref = np.asarray(ours, dtype=np.float64), np.asarray(ref, dtype=np.float64)
    test.assertEqual(ours.shape, ref.shape)
    rel = np.abs(ours - ref).max() / np.abs(ref).max()
    test.assertLess(rel, REL_TOL, f"relative max abs difference {rel:.3e}")


class MaskTests(unittest.TestCase):
    """encoder_masks matches upstream create_4d_mask (bidirectional)."""

    def test_masks_equal_upstream(self):
        """Full and sliding masks, with and without padding, are identical where defined."""
        key_mask = np.array([[1] * 12, [1] * 7 + [0] * 5])
        for km in (key_mask, None):
            ours = encoder_masks(12, None if km is None else mx.array(km), 4)
            t_km = None if km is None else torch.tensor(km)
            ref_full = up.create_4d_mask(12, torch.float32, "cpu", t_km, None, False, False)
            ref_slide = up.create_4d_mask(12, torch.float32, "cpu", t_km, 4, True, False)
            if ours["full_attention"] is None:
                self.assertTrue(bool((ref_full == 0).all()))
            else:
                np.testing.assert_array_equal(
                    np.broadcast_to(np.array(ours["full_attention"]), ref_full.shape),
                    ref_full.numpy(),
                )
            np.testing.assert_array_equal(
                np.broadcast_to(np.array(ours["sliding_attention"]), ref_slide.shape),
                ref_slide.numpy(),
            )


class ConditionEncoderTests(unittest.TestCase):
    """MLXConditionEncoder against AceStepConditionEncoder."""

    def run_both(self, config, lyric_mask, text_mask, order):
        """Same random inputs through both encoders; returns (ours, ref) pairs."""
        torch_enc = randomized(up.AceStepConditionEncoder(config), seed=1)
        ours_enc = mlx_twin(MLXConditionEncoder(config), torch_enc)
        gen = torch.Generator().manual_seed(2)
        b = lyric_mask.shape[0]
        text = torch.randn(b, text_mask.shape[1], config.text_hidden_dim, generator=gen)
        lyric = torch.randn(b, lyric_mask.shape[1], config.text_hidden_dim, generator=gen)
        refer = torch.randn(len(order), 9, config.timbre_hidden_dim, generator=gen)
        t_lm, t_tm, t_order = (torch.tensor(x) for x in (lyric_mask, text_mask, order))
        with torch.no_grad():
            ref_h, ref_m = torch_enc(text, t_tm, lyric, t_lm, refer, t_order)
        h, m = ours_enc(mx.array(text.numpy()), text_mask, mx.array(lyric.numpy()), lyric_mask,
                        mx.array(refer.numpy()), order)
        return (np.array(h), ref_h.numpy()), (m, ref_m.numpy())

    def test_padded_batch_matches(self):
        """Padded lyrics and text, two references for item 0: states and mask match."""
        lyric_mask = np.array([[1] * 10, [1] * 6 + [0] * 4])
        text_mask = np.array([[1] * 5 + [0] * 2, [1] * 7])
        (h, ref_h), (m, ref_m) = self.run_both(tiny_config(), lyric_mask, text_mask,
                                               np.array([0, 0, 1]))
        np.testing.assert_array_equal(m, ref_m)
        assert_close(self, h, ref_h)

    def test_layer_cap_matches(self):
        """num_hidden_layers below the lyric layer count caps the layers run, as upstream."""
        config = tiny_config(num_hidden_layers=1, layer_types=["sliding_attention"] * 3)
        (h, ref_h), _ = self.run_both(config, np.ones((1, 8), np.int64),
                                      np.ones((1, 3), np.int64), np.array([0]))
        assert_close(self, h, ref_h)


class AudioTokenTests(unittest.TestCase):
    """MLX tokenizer, FSQ and detokenizer against upstream."""

    @classmethod
    def setUpClass(cls):
        """One random tokenizer and detokenizer pair shared by the tests."""
        cls.config = tiny_config()
        cls.torch_tok = randomized(up.AceStepAudioTokenizer(cls.config), seed=3)
        cls.ours_tok = mlx_twin(MLXAudioTokenizer(cls.config), cls.torch_tok)
        cls.torch_detok = randomized(up.AudioTokenDetokenizer(cls.config), seed=4)
        cls.ours_detok = mlx_twin(MLXDetokenizer(cls.config), cls.torch_detok)

    def test_codebook_codes_bit_identical(self):
        """Every one of the 64000 indices decodes to exactly upstream's implicit codebook."""
        idx = np.arange(64000).reshape(1, -1, 1)
        ours = np.array(self.ours_tok.quantizer.codes_from_indices(mx.array(idx)))
        ref = self.torch_tok.quantizer.layers[0].implicit_codebook.numpy()
        np.testing.assert_array_equal(ours[0], ref)

    def test_output_from_indices_matches(self):
        """Index decoding including project_out."""
        idx = np.random.default_rng(5).integers(0, 64000, size=(2, 11, 1))
        ours = np.array(self.ours_tok.quantizer.output_from_indices(mx.array(idx)))
        with torch.no_grad():
            ref = self.torch_tok.quantizer.get_output_from_indices(torch.tensor(idx)).numpy()
        assert_close(self, ours, ref)

    def test_tokenizer_matches(self):
        """Pooling and quantization: indices identical, quantized vectors close."""
        x = torch.randn(2, 7, self.config.pool_window_size, self.config.audio_acoustic_hidden_dim,
                        generator=torch.Generator().manual_seed(6))
        with torch.no_grad():
            ref_q, ref_idx = self.torch_tok(x)
        q, idx = self.ours_tok(mx.array(x.numpy()))
        np.testing.assert_array_equal(np.array(idx), ref_idx.numpy())
        assert_close(self, np.array(q), ref_q.numpy())

    def test_detokenizer_matches(self):
        """Window expansion, layers and projection."""
        x = torch.randn(2, 9, self.config.hidden_size, generator=torch.Generator().manual_seed(7))
        with torch.no_grad():
            ref = self.torch_detok(x).numpy()
        assert_close(self, np.array(self.ours_detok(mx.array(x.numpy()))), ref)


if __name__ == "__main__":
    unittest.main()
