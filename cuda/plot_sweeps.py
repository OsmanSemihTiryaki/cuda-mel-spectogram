"""Plot the sweep CSVs into figures for the report.

Produces:
  fig_length.png  : mel-multiply time vs audio length (naive vs tiled), log-log,
                    plus a second panel of tiled-vs-naive speedup.
  fig_tile.png    : mel-multiply time vs tile size.

Usage:
  python plot_sweeps.py --length sweep_length.csv --tile sweep_tile.csv
"""
from __future__ import annotations

import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def fnum(s):
    return float(s) if s not in ("", "NA", None) else None


def plot_length(path, out):
    rows = read_csv(path)
    L   = [fnum(r["length_s"]) for r in rows]
    nv  = [fnum(r["naive_mel_ms"]) for r in rows]
    tl  = [fnum(r["tiled_mel_ms"]) for r in rows]
    cpu = [fnum(r["cpu_ms"]) for r in rows]
    sp  = [fnum(r["mel_speedup"]) for r in rows]

    have_mel = any(v is not None for v in nv) and any(v is not None for v in tl)
    if not have_mel:
        print(f"[skip] {out}: no mel timing data in {path} "
              f"(did the GPU binary run?)")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.plot(L, nv, "o-", label="naive mel")
    ax1.plot(L, tl, "s-", label="tiled+fused mel")
    if any(c is not None for c in cpu):
        ax1.plot(L, cpu, "^--", color="gray", label="CPU reference (full pipeline)")
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("audio length (s)")
    ax1.set_ylabel("time (ms)")
    ax1.set_title("Mel multiply time vs audio length")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend()

    ax2.plot(L, sp, "d-", color="C2")
    ax2.axhline(1.0, color="gray", ls=":", lw=1)
    ax2.set_xscale("log")
    ax2.set_xlabel("audio length (s)")
    ax2.set_ylabel("tiled speedup over naive (x)")
    ax2.set_title("Tiling speedup vs problem size")
    ax2.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"-> {out}")


def plot_tile(path, out):
    rows = read_csv(path)
    T   = [int(r["tile"]) for r in rows]
    mel = [fnum(r["mel_ms"]) for r in rows]

    if not any(v is not None for v in mel):
        print(f"[skip] {out}: no mel timing data in {path}")
        return

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.plot(T, mel, "o-")
    ax.set_xticks(T)
    ax.set_xlabel("tile size T")
    ax.set_ylabel("mel multiply time (ms)")
    ax.set_title("Mel multiply time vs tile size")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"-> {out}")


def plot_mem(path, out):
    rows = read_csv(path)
    modes = [r["mem_mode"] for r in rows]
    h2d = [fnum(r["h2d_ms"]) or 0 for r in rows]
    d2h = [fnum(r["d2h_ms"]) or 0 for r in rows]
    e2e = [fnum(r["e2e_ms"]) or 0 for r in rows]

    import numpy as np
    x = np.arange(len(modes))
    w = 0.6

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

    # transfer cost (H2D + D2H) stacked
    ax1.bar(x, h2d, w, label="H2D")
    ax1.bar(x, d2h, w, bottom=h2d, label="D2H")
    ax1.set_xticks(x); ax1.set_xticklabels(modes)
    ax1.set_ylabel("transfer time (ms)")
    ax1.set_title("Host-device transfer cost by memory mode")
    ax1.legend()
    ax1.grid(True, axis="y", alpha=0.3)

    # end-to-end
    ax2.bar(x, e2e, w, color="C2")
    ax2.set_xticks(x); ax2.set_xticklabels(modes)
    ax2.set_ylabel("end-to-end time (ms)")
    ax2.set_title("End-to-end time by memory mode")
    ax2.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"-> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", default="sweep_length.csv")
    ap.add_argument("--tile", default="sweep_tile.csv")
    ap.add_argument("--mem", default="sweep_mem.csv")
    ap.add_argument("--out-length", default="fig_length.png")
    ap.add_argument("--out-tile", default="fig_tile.png")
    ap.add_argument("--out-mem", default="fig_mem.png")
    args = ap.parse_args()
    plot_length(args.length, args.out_length)
    plot_tile(args.tile, args.out_tile)
    import os
    if os.path.exists(args.mem):
        plot_mem(args.mem, args.out_mem)


if __name__ == "__main__":
    main()
