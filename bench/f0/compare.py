"""Compare generated FLACs against a reference: waveform SNR, correlation, log-spectral distance."""
import glob
import sys

import numpy as np
import soundfile as sf


def load(tag: str) -> np.ndarray:
    """Load the single FLAC produced by run <tag> as mono float64."""
    files = glob.glob(f"out/{tag}/*.flac")
    assert len(files) == 1, (tag, files)
    x, sr = sf.read(files[0])
    assert sr == 48000
    return x.mean(1)


def logspec(x: np.ndarray, n: int = 2048, hop: int = 512) -> np.ndarray:
    """Log-magnitude STFT in dB."""
    w = np.hanning(n)
    frames = np.lib.stride_tricks.sliding_window_view(x, n)[::hop] * w
    return 20 * np.log10(np.abs(np.fft.rfft(frames, axis=1)) + 1e-6)


ref_tag = sys.argv[1]
ref = load(ref_tag)
R = logspec(ref)
for tag in sys.argv[2:]:
    x = load(tag)
    m = min(len(x), len(ref))
    err = x[:m] - ref[:m]
    snr = float("inf") if not err.any() else 10 * np.log10((ref[:m] ** 2).sum() / (err ** 2).sum())
    corr = float(np.corrcoef(x[:m], ref[:m])[0, 1])
    X = logspec(x[:m])
    k = min(len(X), len(R))
    lsd = float(np.sqrt(((X[:k] - R[:k]) ** 2).mean(axis=1)).mean())
    rms = 20 * np.log10(np.sqrt((x ** 2).mean()) + 1e-12)
    print(f"{tag:6s} vs {ref_tag}: SNR {snr:7.2f} dB  corr {corr:.4f}  LSD {lsd:6.2f} dB  rms {rms:6.1f} dBFS")
