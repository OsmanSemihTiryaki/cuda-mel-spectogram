// CUDA implementation of the AST mel-spectrogram pipeline.
//
// The mel-multiply stage has two selectable implementations:
//   - naive: a separate power kernel writing to global memory, followed by a
//     naive GEMM with one thread per output cell and the full K reduction read
//     straight from global memory (the baseline the optimizations beat);
//   - tiled+fused: a single kernel that streams tiles of the power spectrum and
//     filterbank through shared memory and computes the power value on load, so
//     the power spectrum is never materialized in global memory.
// The kernel is chosen at runtime via mel_context_set_kernel().
//
// The audio input and log-mel output buffers can be placed in three ways
// (cudaMalloc + copy, mapped zero-copy, or managed unified memory) selected by
// the MemMode passed to mel_context_create(). The framing, FFT, and log stages
// are the same in all configurations.
//
// The pipeline exposes the same compute_log_mel() signature as the CPU
// reference so the two can be swapped behind one interface. It also exposes a
// persistent context API so that allocations and the cuFFT plan are created
// once and reused across many calls (needed for fair benchmarking).
//
// Layout convention: every multi-dim array is row-major, matching the CPU side.

#pragma once

#include <cstddef>

#include "mel_config.h"

// Host-side memory placement strategy for the audio input and log-mel output.
// On the Jetson's unified memory all three are backed by the same physical
// LPDDR5, so the comparison isolates the cost of the transfer mechanism itself.
enum MemMode {
    MEM_MALLOC  = 0,  // cudaMalloc + cudaMemcpy (mimics a discrete GPU)
    MEM_MAPPED  = 1,  // cudaHostAlloc(cudaHostAllocMapped), zero-copy
    MEM_MANAGED = 2   // cudaMallocManaged, driver-migrated unified memory
};

// ---------------------------------------------------------------------------
// One-shot API (mirrors the CPU reference exactly).
// Allocates, computes, frees. Convenient but not what we benchmark, since it
// pays allocation and plan-creation cost on every call.
// ---------------------------------------------------------------------------
void compute_log_mel_cuda(
    const float*     audio,
    size_t           n_samples,
    const float*     hann_window,
    const float*     mel_filterbank,
    const MelConfig& cfg,
    float*           log_mel
);

// ---------------------------------------------------------------------------
// Persistent-context API (used for benchmarking).
// Create the context once for a fixed (n_samples, cfg), then call run() many
// times. All device buffers and the cuFFT plan live in the context.
// ---------------------------------------------------------------------------
struct MelContext;  // opaque

// Build a context for a fixed input length and configuration.
// hann_window and mel_filterbank are copied to the device here, once.
// mem_mode selects how the audio input and log-mel output buffers are placed
// (see MemMode). Defaults to MEM_MALLOC, the discrete-GPU-style path.
MelContext* mel_context_create(
    size_t           n_samples,
    const float*     hann_window,
    const float*     mel_filterbank,
    const MelConfig& cfg,
    MemMode          mem_mode = MEM_MALLOC
);

// Select the mel-multiply implementation.
//   use_tiled = false -> naive: separate power kernel + naive GEMM
//   use_tiled = true  -> fused tiled kernel (power computed on shared-mem load)
// tile_size is only used when use_tiled is true; must be 8, 16, or 32.
void mel_context_set_kernel(MelContext* ctx, bool use_tiled, int tile_size);

// Run the full pipeline once. `audio` is host memory of length n_samples;
// `log_mel` is host memory of length n_frames * n_mels (caller-allocated).
// How the data crosses the host-device boundary depends on the context's
// MemMode: an explicit copy in MALLOC mode, or direct shared-memory access in
// the mapped and managed modes.
void mel_context_run(
    MelContext*  ctx,
    const float* audio,
    float*       log_mel
);

// Per-stage timing breakdown, in milliseconds. Each field is the time for one
// run() call. Use mel_context_run_profiled() to fill it.
struct MelTimings {
    float h2d;       // host-to-device staging of the audio
    float k1_frame;  // Kernel 1: frame + window
    float fft;       // cuFFT R2C
    float k2_power;  // mel-multiply stage (naive: power + GEMM; tiled: fused)
    float k3_mel;    // unused in the merged timing (kept for struct stability)
    float k4_log;    // Kernel: log compression
    float d2h;       // device-to-host retrieval of the result
    float total;     // sum of the above (device-measured)
};

// Same as mel_context_run() but places CUDA events around every stage and
// returns the per-stage breakdown in `out`. Slightly higher overhead than
// run() because of the extra event records and the per-stage serialization,
// so use run() for the headline end-to-end number and this for the breakdown.
void mel_context_run_profiled(
    MelContext*  ctx,
    const float* audio,
    float*       log_mel,
    MelTimings*  out
);

// Free all device resources held by the context.
void mel_context_destroy(MelContext* ctx);
