"""Final audio files: the raw float WAV upstream writes, or an MP3 made from it here.

MP3 is encoded by libsndfile (bundled with ``soundfile``), so no ffmpeg is needed:
upstream's MP3 path requires an ffmpeg executable, which a Finder-launched app does
not find, and on that failure upstream still reports success with an empty path.
"""

import os

import numpy as np

PEAK_DB = -1.0


def require_audio(path: str) -> str:
    """``path`` if it is a non-empty file.

    Raises:
        RuntimeError: Otherwise (upstream can report success without a file).
    """
    if not path or not os.path.isfile(path) or os.path.getsize(path) == 0:
        raise RuntimeError(f"no audio file was written ({path!r})")
    return path


def peak_normalize(audio: np.ndarray, peak_db: float = PEAK_DB) -> np.ndarray:
    """Scale so the largest absolute sample is ``peak_db`` dBFS; silence is returned as is."""
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak == 0.0:
        return audio
    return (audio * (10 ** (peak_db / 20) / peak)).astype(audio.dtype)


def export_mp3(wav_path: str) -> str:
    """Write ``<same name>.mp3`` (320 kbps CBR, peak at -1 dBFS) and remove the WAV.

    Returns:
        The MP3 path.
    """
    import soundfile as sf

    audio, rate = sf.read(wav_path, dtype="float32", always_2d=True)
    mp3_path = os.path.splitext(wav_path)[0] + ".mp3"
    sf.write(mp3_path, peak_normalize(audio), rate, format="MP3", subtype="MPEG_LAYER_III",
             bitrate_mode="CONSTANT", compression_level=0.0)
    require_audio(mp3_path)
    os.remove(wav_path)
    return mp3_path
