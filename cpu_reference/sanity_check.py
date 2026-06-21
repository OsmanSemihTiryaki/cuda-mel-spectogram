"""Sanity check: run CPU C++ pipeline and compare its output to librosa.

Generates: test audio, Hann window, mel filterbank.
Runs: ./cpu_reference, then librosa equivalent.
Compares:  max-abs-error and mean-abs-error.
"""
import subprocess
import sys
from pathlib import Path

import librosa
import numpy as np

HERE = Path(__file__).parent
RATE = 16000
WINDOW_S = 1.0
N_SAMPLES = int(RATE * WINDOW_S)
FRAME_LEN = 400
HOP = 160
N_FFT = 512
N_MELS = 128
LOG_EPS = 1e-10

# Generate deterministic test audio: 440 Hz sine + light noise
rng = np.random.default_rng(0)
t = np.arange(N_SAMPLES) / RATE
audio = (0.5 * np.sin(2 * np.pi * 440.0 * t) +
         0.05 * rng.standard_normal(N_SAMPLES)).astype(np.float32)
audio.tofile("audio.bin")

# Periodic Hann window
hann = np.hanning(FRAME_LEN + 1)[:-1].astype(np.float32)
hann.tofile("hann.bin")

# Mel filterbank, transposed to [n_bins, n_mels]
fb = librosa.filters.mel(sr=RATE, n_fft=N_FFT, n_mels=N_MELS,
                          norm="slaney", htk=False).astype(np.float32)
np.ascontiguousarray(fb.T).tofile("filterbank.bin")

# Run the C++ pipeline
result = subprocess.run(
    ["./cpu_reference", "audio.bin", "hann.bin", "filterbank.bin",
     "output.bin", "--runs", "20"],
    cwd=HERE, capture_output=True, text=True, check=True,
)
print(result.stdout)

# Load C++ output
n_frames = (N_SAMPLES - FRAME_LEN) // HOP + 1
cpp_out = np.fromfile(HERE / "output.bin", dtype=np.float32).reshape(n_frames, N_MELS)

# Compute reference with the same operations using librosa primitives
# Frame the audio
frames = librosa.util.frame(audio, frame_length=FRAME_LEN, hop_length=HOP, axis=0)
# Window
windowed = frames * hann
# Zero-pad to FFT size
padded = np.zeros((frames.shape[0], N_FFT), dtype=np.float32)
padded[:, :FRAME_LEN] = windowed
# FFT, power
fft_out = np.fft.rfft(padded, axis=1)
power = (fft_out.real ** 2 + fft_out.imag ** 2).astype(np.float32)
# Mel multiply (using same filterbank layout)
mel = power @ fb.T
# Log10 with eps clamp
ref_out = 10.0 * np.log10(np.maximum(mel, LOG_EPS))

# Compare
diff = np.abs(cpp_out - ref_out)
print(f"C++ output shape: {cpp_out.shape}")
print(f"Reference shape: {ref_out.shape}")
print(f"Max abs error: {diff.max():.6e}")
print(f"Mean abs error: {diff.mean():.6e}")
print(f"Output value range: [{cpp_out.min():.3f}, {cpp_out.max():.3f}]")

if diff.max() < 1e-3:
    print("PASS: C++ matches librosa reference within tolerance")
else:
    print("FAIL: numerical mismatch exceeds tolerance")
    sys.exit(1)
