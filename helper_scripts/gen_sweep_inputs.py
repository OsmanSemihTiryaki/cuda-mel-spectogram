"""Generate test inputs for the parameter sweeps.

Writes:
  filterbank.bin, hann.bin (static, config-dependent)
  audio_<ms>.bin (one per requested length, float32 PCM)

The audio is deterministic (fixed seed) so runs are reproducible and the CPU
reference and GPU implementation see identical input.

Usage:
  python gen_sweep_inputs.py --out-dir sweep_data \
      --lengths 0.5 1 2 5 10
"""
from __future__ import annotations

import argparse
import os

import librosa
import numpy as np

RATE = 16000
FRAME = 400
NFFT = 512
NMELS = 128


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="sweep_data")
    ap.add_argument("--lengths", type=float, nargs="+",
                    default=[0.5, 1.0, 2.0, 5.0, 10.0],
                    help="audio lengths in seconds")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Static inputs.
    hann = np.hanning(FRAME + 1)[:-1].astype(np.float32)
    hann.tofile(os.path.join(args.out_dir, "hann.bin"))

    fb = librosa.filters.mel(sr=RATE, n_fft=NFFT, n_mels=NMELS,
                              norm="slaney", htk=False).astype(np.float32)
    np.ascontiguousarray(fb.T).tofile(os.path.join(args.out_dir, "filterbank.bin"))
    print(f"filterbank.bin: ({NFFT//2+1}, {NMELS})  hann.bin: ({FRAME},)")

    # Audio of each length. Use one long deterministic signal and slice it so
    # shorter lengths are prefixes of longer ones (nice property, not required).
    rng = np.random.default_rng(args.seed)
    max_len = int(RATE * max(args.lengths))
    t = np.arange(max_len) / RATE
    full = (0.5 * np.sin(2 * np.pi * 440.0 * t) +
            0.05 * rng.standard_normal(max_len)).astype(np.float32)

    for L in args.lengths:
        n = int(RATE * L)
        ms = int(round(L * 1000))
        path = os.path.join(args.out_dir, f"audio_{ms}.bin")
        full[:n].tofile(path)
        n_frames = (n - FRAME) // 160 + 1
        print(f"audio_{ms}.bin: {n} samples ({L}s) -> {n_frames} frames")


if __name__ == "__main__":
    main()
