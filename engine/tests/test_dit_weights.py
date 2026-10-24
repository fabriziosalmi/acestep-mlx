"""Unit tests for decoder key selection and layout remapping."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from acestep_mlx.dit_weights import remap_decoder_key, select_decoder_weights  # noqa: E402


class RemapDecoderKeyTests(unittest.TestCase):
    """remap_decoder_key mirrors upstream dit_convert naming and layouts."""

    def test_plain_key_is_unchanged(self):
        """Ordinary decoder weights keep their name and get no transform."""
        self.assertEqual(remap_decoder_key("layers.0.self_attn.q_proj.weight"),
                         ("layers.0.self_attn.q_proj.weight", None))

    def test_rotary_buffers_are_skipped(self):
        """Rotary buffers are recomputed in MLX, so they are dropped."""
        self.assertEqual(remap_decoder_key("rotary_emb.inv_freq"), (None, None))

    def test_proj_in_weight_moves_kernel_axis(self):
        """Conv1d weight goes from [out, in, K] to [out, K, in]."""
        key, fn = remap_decoder_key("proj_in.1.weight")
        self.assertEqual(key, "proj_in.weight")
        a = np.arange(2 * 3 * 4).reshape(2, 3, 4)
        np.testing.assert_array_equal(fn(a), a.swapaxes(1, 2))

    def test_proj_out_weight_goes_out_k_in(self):
        """ConvTranspose1d weight goes from [in, out, K] to [out, K, in]."""
        key, fn = remap_decoder_key("proj_out.1.weight")
        self.assertEqual(key, "proj_out.weight")
        a = np.arange(2 * 3 * 4).reshape(2, 3, 4)
        out = fn(a)
        self.assertEqual(out.shape, (3, 4, 2))
        self.assertEqual(out[1, 2, 0], a[0, 1, 2])

    def test_conv_bias_is_renamed_only(self):
        """Biases of the wrapped convs are renamed but not transposed."""
        self.assertEqual(remap_decoder_key("proj_in.1.bias"), ("proj_in.bias", None))
        self.assertEqual(remap_decoder_key("proj_out.1.bias"), ("proj_out.bias", None))


class SelectDecoderWeightsTests(unittest.TestCase):
    """select_decoder_weights keeps only decoder entries of a full state dict."""

    def test_keeps_decoder_entries_only_and_strips_prefix(self):
        """Encoder and tokenizer weights are ignored; decoder names lose the prefix."""
        w = np.zeros((2, 3, 4))
        items = [
            ("encoder.layers.0.weight", np.zeros(1)),
            ("decoder.norm.weight", np.ones(2)),
            ("decoder.proj_in.1.weight", w),
            ("decoder.rotary_emb.inv_freq", np.ones(1)),
            ("tokenizer.x", np.zeros(1)),
        ]
        out = select_decoder_weights(items)
        self.assertEqual([k for k, _ in out], ["norm.weight", "proj_in.weight"])
        self.assertEqual(out[1][1].shape, (2, 4, 3))

    def test_empty_input_gives_empty_output(self):
        """No decoder keys means nothing to load."""
        self.assertEqual(select_decoder_weights([("encoder.a", np.zeros(1))]), [])


if __name__ == "__main__":
    unittest.main()
