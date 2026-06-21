"""Precompute mel filterbank and Hann window as float32 binary files.

These binaries are consumed by both the CPU C++ reference and the CUDA kernels,
so both implementations operate on byte-identical static data.

Outputs (default):
  filterbank.bin : row-major [n_bins, n_mels] = [257, 128] float32, ~128 KB
  hann.bin : float32 array, length frame_length = 400, ~1.6 KB

Mel filterbank uses librosa defaults (slaney scale, slaney norm). This is the
"core" mel definition to validate against. Bit-exact match with HF's Kaldi-style
features is not the goal — Want a valid, standard log-mel spectrogram that
the GPU implementation can reproduce.
"""
from __future__ import annotations

import argparse

import librosa
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate", type=int, default=16000)
    parser.add_argument("--frame-length", type=int, default=400)
    parser.add_argument("--fft-size", type=int, default=512)
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--filterbank-out", type=str, default="filterbank.bin")
    parser.add_argument("--hann-out", type=str, default="hann.bin")
    args = parser.parse_args()

    n_bins = args.fft_size // 2 + 1

    # librosa returns [n_mels, n_bins]; we transpose to [n_bins, n_mels] so that
    # mel[f, m] = sum_k power[f, k] * filterbank[k, m] in our matmul layout.
    fb = librosa.filters.mel(
        sr=args.rate,
        n_fft=args.fft_size,
        n_mels=args.n_mels,
        norm="slaney",
        htk=False,
    ).astype(np.float32)
    fb_t = np.ascontiguousarray(fb.T)  # [n_bins, n_mels]
    fb_t.tofile(args.filterbank_out)
    print(f"Filterbank: shape={fb_t.shape} dtype={fb_t.dtype} "
          f"size={fb_t.nbytes} bytes -> {args.filterbank_out}")

    # Periodic Hann window (matches scipy/librosa STFT convention).
    hann = np.hanning(args.frame_length + 1)[:-1].astype(np.float32)
    hann.tofile(args.hann_out)
    print(f"Hann window: shape={hann.shape} dtype={hann.dtype} "
          f"size={hann.nbytes} bytes -> {args.hann_out}")


if __name__ == "__main__":
    main()
