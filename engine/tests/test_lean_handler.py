"""The lean handler checks model folders without ever starting upstream's download."""

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from acestep.core.generation.handler import init_service_downloads  # noqa: E402

from acestep_mlx.lean_handler import LeanAceStepHandler  # noqa: E402


class EnsureModelsPresentTests(unittest.TestCase):
    """``_ensure_models_present`` with the engine's own set of folders."""

    def setUp(self):
        """A checkpoints folder with DiT, VAE and text encoder weights and no 1.7B LM."""
        self.root = Path(tempfile.mkdtemp())
        for name, weights in (("acestep-v15-turbo", "model.safetensors"),
                              ("vae", "diffusion_pytorch_model.safetensors"),
                              ("Qwen3-Embedding-0.6B", "model.safetensors")):
            (self.root / name).mkdir()
            (self.root / name / weights).write_bytes(b"w")
        self.handler = LeanAceStepHandler()

    def tearDown(self):
        """Remove the folder."""
        shutil.rmtree(self.root)

    def check(self):
        """Run the check with every upstream download replaced by a failure."""
        boom = mock.Mock(side_effect=AssertionError("upstream download started"))
        with mock.patch.object(init_service_downloads, "ensure_main_model", boom), \
                mock.patch.object(init_service_downloads, "ensure_dit_model", boom), \
                mock.patch.object(init_service_downloads, "ensure_vae_model", boom):
            return self.handler._ensure_models_present(
                checkpoint_path=self.root, config_path="acestep-v15-turbo", prefer_source=None,
                vae_variant="official")

    def test_present_without_the_unused_lm_passes_and_downloads_nothing(self):
        """The 1.7B LM is not needed and its absence starts no download."""
        self.assertIsNone(self.check())

    def test_a_missing_folder_is_an_error_not_a_download(self):
        """Without the VAE the check fails with a message naming it."""
        shutil.rmtree(self.root / "vae")
        message, ok = self.check()
        self.assertFalse(ok)
        self.assertIn("vae", message)


if __name__ == "__main__":
    unittest.main()
