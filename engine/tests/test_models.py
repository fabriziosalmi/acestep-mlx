"""Unit tests for the first-start model download, with a fake repository (no network)."""

import hashlib
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from acestep_mlx import models  # noqa: E402
from acestep_mlx.models import ModelError, RemoteFile, Source  # noqa: E402

WEIGHTS = b"w" * 5000
CONFIG = b'{"a": 1}'
SOURCE = Source("dit", "org/repo", "abc123", ("dit/*",))
LM = Source("lm", "org/lm", "def456", subdir="lm")


class FakeRepo:
    """Serves fixed files; ``corrupt`` writes one wrong byte into the weights."""

    def __init__(self, corrupt=False):
        """Start with no downloads."""
        self.corrupt = corrupt
        self.fetched = []

    def list_files(self, source):
        """A weights file with its sha256 (LFS) and a config without one."""
        return [RemoteFile(f"{source.subdir or 'dit'}/model.safetensors", len(WEIGHTS),
                           hashlib.sha256(WEIGHTS).hexdigest()),
                RemoteFile(f"{source.subdir or 'dit'}/config.json", len(CONFIG))]

    def download(self, source, target):
        """Write the files the way snapshot_download does: under ``target``."""
        self.fetched.append(source.name)
        for f in self.list_files(source):
            path = os.path.join(target, f.path)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            data = CONFIG if f.path.endswith(".json") else WEIGHTS
            if self.corrupt and f.sha256:
                data = b"x" + data[1:]
            with open(path, "wb") as fh:
                fh.write(data)


class EnsureTests(unittest.TestCase):
    """``models.ensure``: download once, check the weights, mark, skip next time."""

    def setUp(self):
        """A fresh checkpoints folder per test."""
        self.root = tempfile.mkdtemp()
        self.reports = []

    def tearDown(self):
        """Remove the folder."""
        shutil.rmtree(self.root)

    def ensure(self, repo, sources=(SOURCE,), free=10**12):
        """Run ensure against ``repo`` with ``free`` bytes on the disk, recording the reports."""
        return models.ensure(self.root, sources, lambda *r: self.reports.append(r),
                             list_files=repo.list_files, download=repo.download, interval=0.01,
                             disk_free=lambda path: free)

    def test_too_little_disk_fails_before_downloading(self):
        """Missing bytes plus the spare must fit; otherwise nothing is fetched."""
        repo = FakeRepo()
        need = len(WEIGHTS) + len(CONFIG) + models.SPARE_BYTES
        with self.assertRaisesRegex(ModelError, "GB free"):
            self.ensure(repo, free=need - 1)
        self.assertEqual(repo.fetched, [])
        self.assertEqual(self.ensure(repo, free=need), ["dit"])

    def test_downloads_checks_and_marks_then_skips(self):
        """First call fetches and ends with a complete report; second fetches nothing."""
        repo = FakeRepo()
        self.assertEqual(self.ensure(repo), ["dit"])
        total = len(WEIGHTS) + len(CONFIG)
        self.assertEqual(self.reports[0], (0, total, False))
        self.assertEqual(self.reports[-1], (total, total, True))
        self.assertTrue(os.path.isfile(models.marker_path(self.root, SOURCE)))
        self.reports.clear()
        self.assertEqual(self.ensure(repo), [])
        self.assertEqual(repo.fetched, ["dit"])
        self.assertEqual(self.reports, [])

    def test_corrupted_weights_fail_and_leave_no_marker(self):
        """A sha256 mismatch raises, nothing is marked and no complete report is sent."""
        with self.assertRaisesRegex(ModelError, "sha256"):
            self.ensure(FakeRepo(corrupt=True))
        self.assertFalse(os.path.exists(models.marker_path(self.root, SOURCE)))
        self.assertNotIn(True, [r[2] for r in self.reports])

    def test_missing_file_fails(self):
        """A file listed by the repository but absent after the download raises."""
        repo = FakeRepo()
        real = repo.download
        repo.download = lambda s, t: (real(s, t), os.remove(os.path.join(t, "dit/config.json")))
        with self.assertRaisesRegex(ModelError, "missing"):
            self.ensure(repo)

    def test_subdir_source_lands_in_its_folder(self):
        """A source with a subdir writes under checkpoints/<subdir>/."""
        self.ensure(FakeRepo(), sources=(SOURCE, LM))
        self.assertTrue(os.path.isfile(os.path.join(self.root, "lm", "lm", "model.safetensors")))
        self.assertTrue(os.path.isfile(models.marker_path(self.root, LM)))

    def test_progress_counts_partial_downloads(self):
        """Finished files and huggingface_hub's .incomplete files both count."""
        files = FakeRepo().list_files(SOURCE)
        partial = os.path.join(self.root, ".cache", "huggingface", "download")
        os.makedirs(partial)
        with open(os.path.join(partial, "x.abc.incomplete"), "wb") as fh:
            fh.write(b"p" * 100)
        os.makedirs(os.path.join(self.root, "dit"))
        with open(os.path.join(self.root, "dit", "config.json"), "wb") as fh:
            fh.write(CONFIG)
        self.assertEqual(models.bytes_present(self.root, files), 100 + len(CONFIG))

    def test_a_preallocated_partial_file_counts_by_blocks_written(self):
        """A sparse 1 GB .incomplete file with 64 KB written counts as about 64 KB."""
        partial = os.path.join(self.root, ".cache", "huggingface", "download")
        os.makedirs(partial)
        with open(os.path.join(partial, "w.abc.incomplete"), "wb") as fh:
            fh.write(b"p" * 65536)
            fh.truncate(1 << 30)
        done = models.bytes_present(self.root, [])
        self.assertGreaterEqual(done, 65536)
        self.assertLess(done, 1 << 20)


class SourceTests(unittest.TestCase):
    """Pinned sources and pattern matching."""

    def test_patterns_match_nested_paths_like_allow_patterns(self):
        """``dir/*`` covers files in subfolders, as fnmatch does in huggingface_hub."""
        self.assertTrue(models.MAIN.wants("vae/diffusion_pytorch_model.safetensors"))
        self.assertTrue(models.MAIN.wants("Qwen3-Embedding-0.6B/sub/tokenizer.json"))
        self.assertFalse(models.MAIN.wants("acestep-5Hz-lm-1.7B/model.safetensors"))
        self.assertFalse(models.MAIN.wants("README.md"))

    def test_presets_lm_has_a_source_and_unknown_lm_fails(self):
        """Every preset's LM can be downloaded; an unknown LM is an error, not a guess."""
        from acestep_mlx.recipe import PRESETS

        for preset in PRESETS.values():
            self.assertEqual(models.sources_for(preset.lm_model)[0], models.MAIN)
        with self.assertRaises(ModelError):
            models.sources_for("acestep-5Hz-lm-4B")


if __name__ == "__main__":
    unittest.main()
