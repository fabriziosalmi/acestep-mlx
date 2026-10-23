"""Unit tests for the final audio files (needs numpy and soundfile: the Mac mini venv)."""

import os
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from acestep_mlx.audio_export import export_mp3, peak_normalize, require_audio  # noqa: E402


class AudioExportTests(unittest.TestCase):
    """MP3 export, normalization and the missing-file guard."""

    def test_require_audio_refuses_missing_and_empty(self):
        """Upstream's empty path on a failed save becomes an error."""
        with tempfile.TemporaryDirectory() as root:
            empty = os.path.join(root, "a.wav")
            open(empty, "w").close()
            for path in ("", None, os.path.join(root, "missing.wav"), empty):
                with self.assertRaises(RuntimeError, msg=path):
                    require_audio(path)

    def test_peak_normalize(self):
        """The loudest sample lands at -1 dBFS, signs kept; silence is untouched."""
        out = peak_normalize(np.array([[0.5, -2.0], [1.0, 0.0]], dtype=np.float32))
        self.assertAlmostEqual(float(np.max(np.abs(out))), 10 ** (-1 / 20), places=6)
        self.assertLess(out[0, 1], 0)
        self.assertEqual(out.dtype, np.float32)
        silence = np.zeros((4, 2), dtype=np.float32)
        np.testing.assert_array_equal(peak_normalize(silence), silence)

    def test_export_mp3_replaces_the_wav(self):
        """A float WAV peaking above 0 dBFS becomes a 48 kHz stereo MP3 that does not clip."""
        with tempfile.TemporaryDirectory() as root:
            wav = os.path.join(root, "song.wav")
            t = np.arange(48000 * 2) / 48000
            tone = (1.6 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
            sf.write(wav, np.stack([tone, tone], axis=1), 48000, subtype="FLOAT")
            mp3 = export_mp3(wav)
            self.assertFalse(os.path.exists(wav))
            info = sf.info(mp3)
            self.assertEqual((info.samplerate, info.channels, info.subtype),
                             (48000, 2, "MPEG_LAYER_III"))
            decoded, _ = sf.read(mp3, dtype="float32")
            self.assertLess(float(np.max(np.abs(decoded))), 1.0)
            self.assertGreater(float(np.max(np.abs(decoded))), 0.8)


if __name__ == "__main__":
    unittest.main()
