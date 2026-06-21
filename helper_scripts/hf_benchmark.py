"""HuggingFace ASTFeatureExtractor CPU benchmark.

Measures end-to-end feature extraction time on synthetic audio of configurable
length. Reports median, p95, p99. Output of this script is the "before" number
that the GPU implementation must beat.

Usage:
    python hf_benchmark.py # 1.0s window, 200 runs
    python hf_benchmark.py --window 5.0  # 5s window
    python hf_benchmark.py --save-audio audio.bin # also dump the test waveform
"""
from __future__ import annotations

import argparse
import time

import numpy as np
from transformers import ASTFeatureExtractor

AST_MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate", type=int, default=16000)
    parser.add_argument("--window", type=float, default=1.0,
                        help="Audio window in seconds")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-audio", type=str, default=None,
                        help="If set, save the generated audio to this binary file (float32)")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    n_samples = int(args.rate * args.window)
    audio = rng.uniform(-1.0, 1.0, size=n_samples).astype(np.float32)

    if args.save_audio:
        audio.tofile(args.save_audio)
        print(f"Saved test audio: {args.save_audio} ({n_samples} samples, float32)")

    fe = ASTFeatureExtractor.from_pretrained(AST_MODEL_ID)

    for _ in range(args.warmup):
        _ = fe(audio, sampling_rate=args.rate, return_tensors="np")

    times_ms: list[float] = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        out = fe(audio, sampling_rate=args.rate, return_tensors="np")
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    arr = np.asarray(times_ms)
    print()
    print("HuggingFace ASTFeatureExtractor benchmark")
    print(f"  audio        : {args.window}s @ {args.rate}Hz ({n_samples} samples)")
    print(f"  output shape : {tuple(out['input_values'].shape)}")
    print(f"  warmup runs  : {args.warmup}")
    print(f"  timed runs   : {args.runs}")
    print("  --- timings (ms) ---")
    print(f"  mean         : {arr.mean():.3f}")
    print(f"  median       : {np.median(arr):.3f}")
    print(f"  p95          : {np.percentile(arr, 95):.3f}")
    print(f"  p99          : {np.percentile(arr, 99):.3f}")
    print(f"  min / max    : {arr.min():.3f} / {arr.max():.3f}")


if __name__ == "__main__":
    main()
