"""Unit tests for the configurable VAE tile plan."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from acestep_mlx.vae_tiling import tile_plan  # noqa: E402


def upstream_plan(latent_frames, chunk=512, overlap=64):
    """Re-derive upstream's loop bounds independently, as the reference."""
    stride = chunk - 2 * overlap
    out, core = [], 0
    while core < latent_frames:
        end = min(core + stride, latent_frames)
        out.append((core, end, max(0, core - overlap), min(latent_frames, end + overlap)))
        core += stride
    return out


class TilePlanTests(unittest.TestCase):
    """tile_plan reproduces upstream windows and covers every frame exactly once."""

    def test_matches_upstream_for_default_sizes(self):
        """With upstream's chunk/overlap the windows are identical."""
        for frames, chunk in ((750, 256), (750, 192), (1500, 512), (2000, 256)):
            self.assertEqual(tile_plan(frames, chunk, 64), upstream_plan(frames, chunk, 64))

    def test_cores_cover_every_frame_once(self):
        """Cores are contiguous, non-overlapping and span the whole input."""
        for frames, chunk, overlap in ((750, 128, 32), (750, 64, 16), (101, 40, 8)):
            plan = tile_plan(frames, chunk, overlap)
            self.assertEqual(plan[0][0], 0)
            self.assertEqual(plan[-1][1], frames)
            for prev, nxt in zip(plan, plan[1:]):
                self.assertEqual(prev[1], nxt[0])

    def test_windows_stay_in_bounds_and_contain_core(self):
        """Each window lies inside the input and contains its core."""
        for core_start, core_end, win_start, win_end in tile_plan(750, 64, 16):
            self.assertTrue(0 <= win_start <= core_start < core_end <= win_end <= 750)

    def test_short_input_is_one_window(self):
        """An input that fits in one tile is decoded whole."""
        self.assertEqual(tile_plan(100, 128, 32), [(0, 100, 0, 100)])

    def test_rejects_chunk_without_core(self):
        """A tile no longer than both overlaps has no core and is refused."""
        with self.assertRaises(ValueError):
            tile_plan(750, 128, 64)


if __name__ == "__main__":
    unittest.main()
