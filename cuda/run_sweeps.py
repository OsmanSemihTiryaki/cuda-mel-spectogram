"""Run the parameter sweeps by driving the compiled mel_cuda (and optionally
cpu_reference) binaries, parse their output, and write tidy CSV files.

This keeps the C++ simple: the binary runs one configuration per invocation,
and this script orchestrates the grid and collects results.

Sweeps produced:
  1. audio length    (naive vs tiled, each length)   -> sweep_length.csv
  2. tile size       (T in 8,16,32 at fixed length)  -> sweep_tile.csv
The naive-vs-tiled comparison falls out of sweep 1 (both kernels at each length).

Usage (from the cuda build dir, with sweep_data/ alongside):
  python run_sweeps.py \
      --mel-cuda ./mel_cuda \
      --cpu-ref ../cpu_reference/cpu_reference \
      --data-dir sweep_data \
      --lengths 0.5 1 2 5 10 \
      --runs 200
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess

# Regexes matching the exact stdout of the binaries.
RE_MEAN     = re.compile(r"mean per run\s*:\s*([\d.]+)")
RE_MEL      = re.compile(r"mel \([^)]*\)\s+([\d.]+)\s*ms")
RE_SUM      = re.compile(r"^\s*sum\s+([\d.]+)\s*ms", re.MULTILINE)
RE_H2D      = re.compile(r"H2D copy\s+([\d.]+)\s*ms")
RE_D2H      = re.compile(r"D2H copy\s+([\d.]+)\s*ms")
RE_MAXERR   = re.compile(r"max abs error\s*:\s*([\d.eE+-]+)")
RE_PASS     = re.compile(r"\b(PASS|FAIL)\b")
RE_CPU_MEAN = re.compile(r"mean per run\s*:\s*([\d.]+)")
RE_NFRAMES  = re.compile(r"output shape\s*:\s*\((\d+),")


def run(cmd: list[str]) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{r.stderr}\n{r.stdout}")
    return r.stdout


def grab(rx: re.Pattern, text: str, cast=float, default=None):
    m = rx.search(text)
    return cast(m.group(1)) if m else default


def fmt(v, spec="10.4f"):
    """Format a number, or right-aligned 'NA' if it is None."""
    if v is None:
        width = spec.split(".")[0].lstrip(">")
        try:
            w = int(width)
        except ValueError:
            w = 10
        return f"{'NA':>{w}}"
    return f"{v:{spec}}"


def run_mel(mel_bin, audio, hann, fb, ref, runs, mode, tile=16, mem="malloc"):
    cmd = [mel_bin, audio, hann, fb, "/tmp/_sweep_out.bin",
           "--runs", str(runs), "--ref", ref, "--mem", mem]
    if mode == "naive":
        cmd.append("--naive")
    else:
        cmd += ["--tile", str(tile)]
    out = run(cmd)
    return {
        "end_to_end_ms": grab(RE_MEAN, out),
        "mel_ms":        grab(RE_MEL, out),
        "sum_ms":        grab(RE_SUM, out),
        "h2d_ms":        grab(RE_H2D, out),
        "d2h_ms":        grab(RE_D2H, out),
        "max_abs_err":   grab(RE_MAXERR, out, cast=float),
        "status":        grab(RE_PASS, out, cast=str, default="?"),
        "n_frames":      grab(RE_NFRAMES, out, cast=int),
    }


def run_cpu(cpu_bin, audio, hann, fb, runs):
    if not cpu_bin or not os.path.exists(cpu_bin):
        return None
    out = run([cpu_bin, audio, hann, fb, "/tmp/_sweep_cpu.bin", "--runs", str(runs)])
    return grab(RE_CPU_MEAN, out)


def sweep_length(args):
    hann = os.path.join(args.data_dir, "hann.bin")
    fb   = os.path.join(args.data_dir, "filterbank.bin")
    rows = []
    print("\n=== Sweep 1: audio length (naive vs tiled) ===")
    print(f"{'len_s':>6} {'frames':>7} {'naive_mel':>10} {'tiled_mel':>10} "
          f"{'speedup':>8} {'naive_e2e':>10} {'tiled_e2e':>10} {'cpu_ms':>9}")
    for L in args.lengths:
        ms = int(round(L * 1000))
        audio = os.path.join(args.data_dir, f"audio_{ms}.bin")
        # CPU reference uses its own output as ground truth; for GPU validation
        # we need a CPU output file at this length. Generate it via cpu-ref.
        ref = f"/tmp/_ref_{ms}.bin"
        if args.cpu_ref and os.path.exists(args.cpu_ref):
            run([args.cpu_ref, audio, hann, fb, ref, "--runs", "1"])
        else:
            ref = "/dev/null"  # validation will be skipped/marked ?

        naive = run_mel(args.mel_cuda, audio, hann, fb, ref, args.runs, "naive")
        tiled = run_mel(args.mel_cuda, audio, hann, fb, ref, args.runs, "tiled", args.tile)
        cpu_ms = run_cpu(args.cpu_ref, audio, hann, fb, max(20, args.runs // 5))

        speedup = (naive["mel_ms"] / tiled["mel_ms"]
                   if naive["mel_ms"] and tiled["mel_ms"] else None)

        print(f"{L:6.1f} {tiled['n_frames'] or 0:7d} "
              f"{fmt(naive['mel_ms'])} {fmt(tiled['mel_ms'])} "
              f"{fmt(speedup, '7.2f')}x "
              f"{fmt(naive['end_to_end_ms'])} {fmt(tiled['end_to_end_ms'])} "
              f"{fmt(cpu_ms, '9.3f')}")

        rows.append({
            "length_s": L,
            "n_frames": tiled["n_frames"],
            "naive_mel_ms": naive["mel_ms"],
            "tiled_mel_ms": tiled["mel_ms"],
            "mel_speedup": speedup,
            "naive_e2e_ms": naive["end_to_end_ms"],
            "tiled_e2e_ms": tiled["end_to_end_ms"],
            "cpu_ms": cpu_ms,
            "naive_status": naive["status"],
            "tiled_status": tiled["status"],
        })

    with open(args.out_length, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"-> wrote {args.out_length}")


def sweep_tile(args):
    hann = os.path.join(args.data_dir, "hann.bin")
    fb   = os.path.join(args.data_dir, "filterbank.bin")
    ms = int(round(args.tile_sweep_length * 1000))
    audio = os.path.join(args.data_dir, f"audio_{ms}.bin")
    ref = f"/tmp/_ref_{ms}.bin"
    if args.cpu_ref and os.path.exists(args.cpu_ref):
        run([args.cpu_ref, audio, hann, fb, ref, "--runs", "1"])
    else:
        ref = "/dev/null"

    rows = []
    print(f"\n=== Sweep 2: tile size (at {args.tile_sweep_length}s) ===")
    print(f"{'tile':>5} {'mel_ms':>10} {'e2e_ms':>10} {'status':>7}")
    for T in (8, 16, 32):
        r = run_mel(args.mel_cuda, audio, hann, fb, ref, args.runs, "tiled", T)
        print(f"{T:5d} {fmt(r['mel_ms'])} {fmt(r['end_to_end_ms'])} {r['status']:>7}")
        rows.append({
            "tile": T,
            "mel_ms": r["mel_ms"],
            "e2e_ms": r["end_to_end_ms"],
            "status": r["status"],
        })

    with open(args.out_tile, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"-> wrote {args.out_tile}")


def sweep_memory(args):
    hann = os.path.join(args.data_dir, "hann.bin")
    fb   = os.path.join(args.data_dir, "filterbank.bin")
    ms = int(round(args.mem_sweep_length * 1000))
    audio = os.path.join(args.data_dir, f"audio_{ms}.bin")
    ref = f"/tmp/_ref_{ms}.bin"
    if args.cpu_ref and os.path.exists(args.cpu_ref):
        run([args.cpu_ref, audio, hann, fb, ref, "--runs", "1"])
    else:
        ref = "/dev/null"

    rows = []
    print(f"\n=== Sweep 3: memory placement (tiled, at {args.mem_sweep_length}s) ===")
    print(f"{'mode':>9} {'h2d_ms':>8} {'d2h_ms':>8} {'e2e_ms':>9} {'status':>7}")
    for mem in ("malloc", "mapped", "managed"):
        r = run_mel(args.mel_cuda, audio, hann, fb, ref, args.runs,
                    "tiled", args.tile, mem)
        print(f"{mem:>9} {fmt(r['h2d_ms'], '8.4f')} {fmt(r['d2h_ms'], '8.4f')} "
              f"{fmt(r['end_to_end_ms'], '9.4f')} {r['status']:>7}")
        rows.append({
            "mem_mode": mem,
            "h2d_ms": r["h2d_ms"],
            "d2h_ms": r["d2h_ms"],
            "e2e_ms": r["end_to_end_ms"],
            "status": r["status"],
        })

    with open(args.out_mem, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"-> wrote {args.out_mem}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mel-cuda", default="./mel_cuda")
    ap.add_argument("--cpu-ref", default="../cpu_reference/cpu_reference")
    ap.add_argument("--data-dir", default="sweep_data")
    ap.add_argument("--lengths", type=float, nargs="+",
                    default=[0.5, 1.0, 2.0, 5.0, 10.0])
    ap.add_argument("--tile", type=int, default=16,
                    help="tile size used in the length sweep")
    ap.add_argument("--tile-sweep-length", type=float, default=10.0,
                    help="audio length for the tile-size sweep")
    ap.add_argument("--mem-sweep-length", type=float, default=10.0,
                    help="audio length for the memory-placement sweep")
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--out-length", default="sweep_length.csv")
    ap.add_argument("--out-tile", default="sweep_tile.csv")
    ap.add_argument("--out-mem", default="sweep_mem.csv")
    args = ap.parse_args()

    sweep_length(args)
    sweep_tile(args)
    sweep_memory(args)
    print("\nAll sweeps complete.")


if __name__ == "__main__":
    main()
