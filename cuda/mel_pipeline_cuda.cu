// CUDA implementation of the AST mel-spectrogram pipeline.
// See mel_pipeline_cuda.h for the design rationale and the public API.
//
// Pipeline stages:
//   audio --(H2D / mapped / managed)--> [K1: frame + window] --> framed (F x N)
//                  [cuFFT R2C batched]                        --> complex (F x K)
//                  [mel multiply]                             --> mel     (F x M)
//                  [K4: log]                                  --> log-mel (F x M)
//                                                             --(D2H / shared)--> host
//
// The mel-multiply stage has two implementations selected at runtime:
//   - naive: a separate power kernel (K2) writes the power spectrum to global
//     memory, then a naive GEMM (K3) reads it back with one thread per output
//     cell. This is the baseline.
//   - tiled+fused: a single kernel streams tiles of the power spectrum and the
//     filterbank through shared memory and computes the power value on load, so
//     the power spectrum is never written to global memory.
//
// The audio input and log-mel output buffers are placed according to MemMode
// (cudaMalloc + copy, mapped zero-copy, or managed unified memory); the framing,
// FFT, and log stages are identical across all configurations.

#include "mel_pipeline_cuda.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <cufft.h>

// --- error checking -------------------------------------------------------
// borrowed pattern from https://leimao.github.io/blog/Proper-CUDA-Error-Checking/
#define CC(call)                                                              \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n",                 \
                         __FILE__, __LINE__, cudaGetErrorString(err));        \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

#define CF(call)                                                              \
    do {                                                                      \
        cufftResult err = (call);                                             \
        if (err != CUFFT_SUCCESS) {                                           \
            std::fprintf(stderr, "cuFFT error at %s:%d: code %d\n",           \
                         __FILE__, __LINE__, (int)err);                       \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

// Host-side mirror of compute_n_frames (kept here to avoid linking the CPU obj).
static size_t n_frames_of(size_t n_samples, int frame_length, int hop_length) {
    if (n_samples < (size_t)frame_length) return 0;
    return (n_samples - (size_t)frame_length) / (size_t)hop_length + 1;
}

// ===========================================================================
// Kernel 1: framing + Hann window + zero-pad.
// One thread per output sample of the framed buffer (shape F x N).
// Naive: the Hann window is read from plain global memory, not constant memory.
// ===========================================================================
__global__ void k1_frame_window(
    const float* __restrict__ audio,
    const float* __restrict__ hann,   // length W, in global memory
    float*       __restrict__ framed, // F x N
    int F, int N, int W, int hop
) {
    // 2D grid: x covers samples within a frame, y covers frames.
    int n = blockIdx.x * blockDim.x + threadIdx.x;  // sample index 0..N-1
    int f = blockIdx.y * blockDim.y + threadIdx.y;  // frame index  0..F-1
    if (f >= F || n >= N) return;

    float value;
    if (n < W) {
        value = audio[(size_t)f * hop + n] * hann[n];
    } else {
        value = 0.0f;  // zero-pad the tail (W..N)
    }
    framed[(size_t)f * N + n] = value;
}

// ===========================================================================
// Kernel 2: power spectrum.
// One thread per complex bin (shape F x K). Naive: writes the full power
// spectrum back to global memory instead of fusing into the GEMM.
// ===========================================================================
__global__ void k2_power(
    const cufftComplex* __restrict__ spec,  // F x K
    float*              __restrict__ power, // F x K
    int F, int K
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = F * K;
    if (idx >= total) return;

    cufftComplex z = spec[idx];
    power[idx] = z.x * z.x + z.y * z.y;
}

// ===========================================================================
// Kernel 3: naive mel filterbank multiply (GEMM).
// Computes S = P * Phi, where P is (F x K) and Phi is (K x M).
// One thread per output cell S[f, m]. The K-element reduction is a plain loop
// reading both operands straight from global memory. No shared memory, no
// tiling. This is the baseline the tiled kernel will be compared against.
// ===========================================================================
__global__ void k3_mel_naive(
    const float* __restrict__ power, // F x K
    const float* __restrict__ phi,   // K x M
    float*       __restrict__ mel,   // F x M
    int F, int K, int M
) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;  // mel bin   0..M-1
    int f = blockIdx.y * blockDim.y + threadIdx.y;  // frame     0..F-1
    if (f >= F || m >= M) return;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += power[(size_t)f * K + k] * phi[(size_t)k * M + m];
    }
    mel[(size_t)f * M + m] = acc;
}

// ===========================================================================
// Kernel 3 (optimized): fused power + tiled mel GEMM.
// Adapted from the tiled sgemm_smem of the lab2 code I have implemented.
//
// Computes the same S = P * Phi as the naive kernel, but:
//   - tiles of P and Phi are streamed through shared memory (the standard
//     tiled-GEMM pattern), so each loaded value is reused TILE times instead
//     of being re-read from global memory for every output cell;
//   - the power spectrum is never materialized in global memory: when a thread
//     loads its P-tile element, it reads the complex FFT output directly and
//     stores the squared magnitude re^2 + im^2 into shared memory. This fuses
//     the old power kernel (K2) into this one and removes a global round-trip.
//
// Mapping onto the SGEMM shape (A is M x K, B is K x N, C is M x N):
//   A  <- spec (the FFT output), logically the power matrix, F x K  -> M = F
//   B  <- phi (the filterbank),                              K x M  -> N = n_mels
//   C  <- mel (the output),                                  F x M  -> M x N
//
// TILE is a template parameter so the host can sweep tile sizes (8, 16, 32)
// without code changes
// ===========================================================================
template <int TILE>
__global__ void k3_mel_tiled_fused(
    const cufftComplex* __restrict__ spec, // F x K, complex (power read on load)
    const float*        __restrict__ phi,  // K x M
    float*              __restrict__ mel,  // F x M
    int F, int K, int M
) {
    __shared__ float P_tile[TILE][TILE];   // holds power values, not complex
    __shared__ float Phi_tile[TILE][TILE];

    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;

    // Global row (frame) and column (mel bin) of the output cell this thread owns.
    int row = blockIdx.y * TILE + threadRow;  // frame index   0..F-1
    int col = blockIdx.x * TILE + threadCol;  // mel bin index 0..M-1

    float acc = 0.0f;

    // Slide the tile window along the shared K dimension.
    for (int t = 0; t < CEIL_DIV(K, TILE); ++t) {
        int pCol = t * TILE + threadCol;   // column into P / spec (a K index)
        int phiRow = t * TILE + threadRow; // row into Phi          (a K index)

        // Load one P-tile element: read the complex FFT value and fuse the
        // power computation (squared magnitude) right here.
        if (row < F && pCol < K) {
            cufftComplex z = spec[(size_t)row * K + pCol];
            P_tile[threadRow][threadCol] = z.x * z.x + z.y * z.y;
        } else {
            P_tile[threadRow][threadCol] = 0.0f;
        }

        // Load one Phi-tile element: plain global-to-shared copy (unchanged
        // from the original sgemm_smem B-tile load).
        if (phiRow < K && col < M) {
            Phi_tile[threadRow][threadCol] = phi[(size_t)phiRow * M + col];
        } else {
            Phi_tile[threadRow][threadCol] = 0.0f;
        }

        __syncthreads();

        // Accumulate the partial dot product from this tile.
        for (int k = 0; k < TILE; ++k) {
            acc += P_tile[threadRow][k] * Phi_tile[k][threadCol];
        }

        __syncthreads();
    }

    if (row < F && col < M) {
        mel[(size_t)row * M + col] = acc;
    }
}

// ===========================================================================
// Kernel 4: log compression.
// One thread per output element (shape F x M). out = 10 * log10(max(s, eps)).
// ===========================================================================
__global__ void k4_log(
    float* __restrict__ mel,  // F x M, in place
    int total, float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    float s = mel[idx];
    s = s > eps ? s : eps;
    mel[idx] = 10.0f * log10f(s);
}

// ===========================================================================
// Persistent context
// ===========================================================================
struct MelContext {
    MelConfig cfg;
    size_t    n_samples;
    size_t    n_frames;
    int       K;  // fft_size/2 + 1

    // kernel selection for the mel multiply stage:
    //   use_tiled = false -> naive K2(power) + K3(naive GEMM)
    //   use_tiled = true  -> single fused tiled kernel (power computed on load)
    // tile_size selects the template instantiation (8, 16, or 32).
    bool use_tiled = true;
    int  tile_size = 16;

    // memory placement for the audio input and log-mel output buffers.
    MemMode mem_mode = MEM_MALLOC;

    // device buffers
    float*        d_audio   = nullptr;  // n_samples (device ptr used by kernels)
    float*        d_hann    = nullptr;  // W
    float*        d_phi     = nullptr;  // K x M
    float*        d_framed  = nullptr;  // F x N
    cufftComplex* d_spec    = nullptr;  // F x K
    float*        d_power   = nullptr;  // F x K (only used in naive path)
    float*        d_mel     = nullptr;  // F x M (device ptr; also log-mel out)

    // Host-accessible views of the input/output buffers. In MALLOC mode these
    // are nullptr (we cudaMemcpy from the caller's buffers instead). In MAPPED
    // and MANAGED mode these point to memory the CPU can write/read directly:
    //   MAPPED  -> pinned mapped host allocation (h_* is host ptr, d_* is the
    //              device ptr obtained via cudaHostGetDevicePointer)
    //   MANAGED -> a single managed pointer (h_* == d_*)
    float*        h_audio   = nullptr;  // n_samples
    float*        h_mel     = nullptr;  // F x M

    cufftHandle   plan;
};

MelContext* mel_context_create(
    size_t           n_samples,
    const float*     hann_window,
    const float*     mel_filterbank,
    const MelConfig& cfg,
    MemMode          mem_mode
) {
    MelContext* ctx = new MelContext();
    ctx->cfg       = cfg;
    ctx->n_samples = n_samples;
    ctx->n_frames  = n_frames_of(n_samples, cfg.frame_length, cfg.hop_length);
    ctx->K         = cfg.fft_size / 2 + 1;
    ctx->mem_mode  = mem_mode;

    const size_t F = ctx->n_frames;
    const int    N = cfg.fft_size;
    const int    K = ctx->K;
    const int    M = cfg.n_mels;
    const int    W = cfg.frame_length;

    // Internal pipeline buffers always live in plain device memory; only the
    // audio input and log-mel output buffers vary with the memory mode.
    CC(cudaMalloc(&ctx->d_hann,   (size_t)W         * sizeof(float)));
    CC(cudaMalloc(&ctx->d_phi,    (size_t)K * M     * sizeof(float)));
    CC(cudaMalloc(&ctx->d_framed, F * N             * sizeof(float)));
    CC(cudaMalloc(&ctx->d_spec,   F * K             * sizeof(cufftComplex)));
    CC(cudaMalloc(&ctx->d_power,  F * K             * sizeof(float)));

    const size_t audio_bytes = n_samples * sizeof(float);
    const size_t mel_bytes   = F * M      * sizeof(float);

    switch (mem_mode) {
        case MEM_MALLOC:
            // Device buffers; run() will cudaMemcpy to/from the caller's host
            // memory. h_audio / h_mel stay null.
            CC(cudaMalloc(&ctx->d_audio, audio_bytes));
            CC(cudaMalloc(&ctx->d_mel,   mel_bytes));
            break;

        case MEM_MAPPED:
            // Pinned, mapped host allocations. The CPU writes/reads h_*, and the
            // kernels use the corresponding device pointer (same physical memory
            // on the Jetson, no explicit copy). Requires cudaDeviceMapHost to be
            // set before CUDA context init (done in main before any CUDA call).
            CC(cudaHostAlloc(&ctx->h_audio, audio_bytes, cudaHostAllocMapped));
            CC(cudaHostAlloc(&ctx->h_mel,   mel_bytes,   cudaHostAllocMapped));
            CC(cudaHostGetDevicePointer(&ctx->d_audio, ctx->h_audio, 0));
            CC(cudaHostGetDevicePointer(&ctx->d_mel,   ctx->h_mel,   0));
            break;

        case MEM_MANAGED:
            // Single managed pointers usable from host and device; the driver
            // migrates pages on demand. h_* and d_* alias the same pointer.
            CC(cudaMallocManaged(&ctx->d_audio, audio_bytes));
            CC(cudaMallocManaged(&ctx->d_mel,   mel_bytes));
            ctx->h_audio = ctx->d_audio;
            ctx->h_mel   = ctx->d_mel;
            break;
    }

    // Static data (Hann window, filterbank) copied once.
    CC(cudaMemcpy(ctx->d_hann, hann_window, (size_t)W * sizeof(float),
                  cudaMemcpyHostToDevice));
    CC(cudaMemcpy(ctx->d_phi, mel_filterbank, (size_t)K * M * sizeof(float),
                  cudaMemcpyHostToDevice));

    // Batched real-to-complex 1D FFT plan: F transforms of length N.
    // Input rows are length N (real), output rows are length K (complex).
    int n[1] = { N };
    CF(cufftPlanMany(
        &ctx->plan,
        /*rank=*/1, n,
        /*inembed=*/nullptr, /*istride=*/1, /*idist=*/N,
        /*onembed=*/nullptr, /*ostride=*/1, /*odist=*/K,
        CUFFT_R2C, /*batch=*/(int)F));

    return ctx;
}

void mel_context_set_kernel(MelContext* ctx, bool use_tiled, int tile_size) {
    ctx->use_tiled = use_tiled;
    ctx->tile_size = tile_size;
}
// Naive path: K2 (power -> global) then K3 (naive GEMM).
// Tiled path: single fused kernel, power computed on load. The tile size
// selects the template instantiation.
static void launch_mel_multiply(MelContext* ctx) {
    const size_t F = ctx->n_frames;
    const int    K = ctx->K;
    const int    M = ctx->cfg.n_mels;

    if (!ctx->use_tiled) {
        // K2: power -> global memory.
        {
            int total = (int)F * K;
            int block = 256;
            int grid  = (total + block - 1) / block;
            k2_power<<<grid, block>>>(ctx->d_spec, ctx->d_power, (int)F, K);
            CC(cudaGetLastError());
        }
        // K3: naive GEMM reading power from global memory.
        {
            dim3 block(16, 16);
            dim3 grid(CEIL_DIV(M, 16), CEIL_DIV((int)F, 16));
            k3_mel_naive<<<grid, block>>>(
                ctx->d_power, ctx->d_phi, ctx->d_mel, (int)F, K, M);
            CC(cudaGetLastError());
        }
        return;
    }

    // Tiled + fused path. Grid covers the output (M columns, F rows) in TILE
    // steps; block is TILE x TILE threads.
    const int T = ctx->tile_size;
    dim3 block(T, T);
    dim3 grid(CEIL_DIV(M, T), CEIL_DIV((int)F, T));
    switch (T) {
        case 8:
            k3_mel_tiled_fused<8><<<grid, block>>>(
                ctx->d_spec, ctx->d_phi, ctx->d_mel, (int)F, K, M);
            break;
        case 16:
            k3_mel_tiled_fused<16><<<grid, block>>>(
                ctx->d_spec, ctx->d_phi, ctx->d_mel, (int)F, K, M);
            break;
        case 32:
            k3_mel_tiled_fused<32><<<grid, block>>>(
                ctx->d_spec, ctx->d_phi, ctx->d_mel, (int)F, K, M);
            break;
        default:
            std::fprintf(stderr, "unsupported tile size %d (use 8, 16, or 32)\n", T);
            std::exit(EXIT_FAILURE);
    }
    CC(cudaGetLastError());
}

// Stage the audio input into device-readable memory according to mem_mode.
//   MALLOC  -> cudaMemcpy H2D into the device buffer
//   MAPPED  -> CPU memcpy into the pinned mapped host buffer (kernel reads it
//              directly via the mapped device pointer; no cudaMemcpy)
//   MANAGED -> CPU memcpy into the managed buffer (driver migrates on access)
static void stage_input(MelContext* ctx, const float* audio) {
    const size_t bytes = ctx->n_samples * sizeof(float);
    if (ctx->mem_mode == MEM_MALLOC) {
        CC(cudaMemcpy(ctx->d_audio, audio, bytes, cudaMemcpyHostToDevice));
    } else {
        std::memcpy(ctx->h_audio, audio, bytes);
    }
}

// Retrieve the log-mel output from device-readable memory according to mem_mode.
static void retrieve_output(MelContext* ctx, float* log_mel) {
    const size_t bytes = ctx->n_frames * (size_t)ctx->cfg.n_mels * sizeof(float);
    if (ctx->mem_mode == MEM_MALLOC) {
        CC(cudaMemcpy(log_mel, ctx->d_mel, bytes, cudaMemcpyDeviceToHost));
    } else {
        // Kernel writes complete after the sync in run(); read host-side memory.
        std::memcpy(log_mel, ctx->h_mel, bytes);
    }
}

void mel_context_run(
    MelContext*  ctx,
    const float* audio,
    float*       log_mel
) {
    const MelConfig& cfg = ctx->cfg;
    const size_t F = ctx->n_frames;
    const int    N = cfg.fft_size;
    const int    K = ctx->K;
    const int    M = cfg.n_mels;
    const int    W = cfg.frame_length;

    // Stage audio input (transfer mechanism depends on mem_mode).
    stage_input(ctx, audio);

    // K1: frame + window. 2D grid (samples, frames).
    {
        dim3 block(256, 1);
        dim3 grid((N + block.x - 1) / block.x, (unsigned)F);
        k1_frame_window<<<grid, block>>>(
            ctx->d_audio, ctx->d_hann, ctx->d_framed, (int)F, N, W, cfg.hop_length);
        CC(cudaGetLastError());
    }

    // cuFFT R2C.
    CF(cufftExecR2C(ctx->plan, ctx->d_framed, ctx->d_spec));

    // Mel multiply (naive two-kernel path or fused tiled path, per ctx).
    launch_mel_multiply(ctx);

    // K4: log compression, in place. 1D grid over F*M.
    {
        int total = (int)F * M;
        int block = 256;
        int grid  = (total + block - 1) / block;
        k4_log<<<grid, block>>>(ctx->d_mel, total, cfg.log_eps);
        CC(cudaGetLastError());
    }

    // Ensure all kernels have completed before the host reads the output.
    CC(cudaDeviceSynchronize());

    // Retrieve log-mel output (transfer mechanism depends on mem_mode).
    retrieve_output(ctx, log_mel);
}

void mel_context_run_profiled(
    MelContext*  ctx,
    const float* audio,
    float*       log_mel,
    MelTimings*  out
) {
    const MelConfig& cfg = ctx->cfg;
    const size_t F = ctx->n_frames;
    const int    N = cfg.fft_size;
    const int    K = ctx->K;
    const int    M = cfg.n_mels;
    const int    W = cfg.frame_length;

    // Eight events delimit seven timed segments: H2D, K1, FFT, K2, K3, K4, D2H.
    cudaEvent_t ev[8];
    for (int i = 0; i < 8; ++i) CC(cudaEventCreate(&ev[i]));

    CC(cudaEventRecord(ev[0]));

    // Input staging. In MALLOC mode this is a device-side async copy (captured
    // by the events). In MAPPED/MANAGED mode there is no device copy; the CPU
    // memcpy in stage_input runs off-stream, so this segment reads ~0, which is
    // the honest signal that no host->device transfer occurs.
    if (ctx->mem_mode == MEM_MALLOC) {
        CC(cudaMemcpyAsync(ctx->d_audio, audio, ctx->n_samples * sizeof(float),
                           cudaMemcpyHostToDevice));
    } else {
        stage_input(ctx, audio);
    }
    CC(cudaEventRecord(ev[1]));

    // K1: frame + window
    {
        dim3 block(256, 1);
        dim3 grid((N + block.x - 1) / block.x, (unsigned)F);
        k1_frame_window<<<grid, block>>>(
            ctx->d_audio, ctx->d_hann, ctx->d_framed, (int)F, N, W, cfg.hop_length);
        CC(cudaGetLastError());
    }
    CC(cudaEventRecord(ev[2]));

    // cuFFT R2C
    CF(cufftExecR2C(ctx->plan, ctx->d_framed, ctx->d_spec));
    CC(cudaEventRecord(ev[3]));

    // Mel multiply: time the whole stage as one segment, whichever path is
    // active. In naive mode this covers power + naive GEMM; in tiled mode it
    // covers the single fused kernel. We report it in the k3_mel field and
    // leave k2_power at zero so the breakdown has one "mel multiply" number
    // that is directly comparable between the two modes.
    launch_mel_multiply(ctx);
    CC(cudaEventRecord(ev[4]));

    // (ev[4]..ev[5] is unused as a separate segment now; record a zero-width
    // marker so the seven-segment bookkeeping below stays valid.)
    CC(cudaEventRecord(ev[5]));

    // K4: log compression
    {
        int total = (int)F * M;
        int block = 256;
        int grid  = (total + block - 1) / block;
        k4_log<<<grid, block>>>(ctx->d_mel, total, cfg.log_eps);
        CC(cudaGetLastError());
    }
    CC(cudaEventRecord(ev[6]));

    // Output retrieval. MALLOC: device-side async copy (timed). MAPPED/MANAGED:
    // ensure kernels finished, then CPU memcpy (off-stream, segment reads ~0).
    if (ctx->mem_mode == MEM_MALLOC) {
        CC(cudaMemcpyAsync(log_mel, ctx->d_mel, F * M * sizeof(float),
                           cudaMemcpyDeviceToHost));
        CC(cudaEventRecord(ev[7]));
        CC(cudaEventSynchronize(ev[7]));
    } else {
        CC(cudaEventRecord(ev[7]));
        CC(cudaEventSynchronize(ev[7]));
        retrieve_output(ctx, log_mel);
    }

    float seg[7];
    for (int i = 0; i < 7; ++i) CC(cudaEventElapsedTime(&seg[i], ev[i], ev[i + 1]));

    out->h2d      = seg[0];
    out->k1_frame = seg[1];
    out->fft      = seg[2];
    out->k2_power = seg[3];  // mel-multiply stage (power + GEMM, or fused)
    out->k3_mel   = seg[4];  // zero (kept for struct stability)
    out->k4_log   = seg[5];
    out->d2h      = seg[6];
    out->total    = seg[0] + seg[1] + seg[2] + seg[3] + seg[4] + seg[5] + seg[6];

    for (int i = 0; i < 8; ++i) CC(cudaEventDestroy(ev[i]));
}

void mel_context_destroy(MelContext* ctx) {
    if (!ctx) return;
    // Internal device buffers.
    cudaFree(ctx->d_hann);
    cudaFree(ctx->d_phi);
    cudaFree(ctx->d_framed);
    cudaFree(ctx->d_spec);
    cudaFree(ctx->d_power);
    // Audio/mel buffers: free according to how they were allocated.
    switch (ctx->mem_mode) {
        case MEM_MALLOC:
            cudaFree(ctx->d_audio);
            cudaFree(ctx->d_mel);
            break;
        case MEM_MAPPED:
            cudaFreeHost(ctx->h_audio);  // frees the mapped pinned allocation
            cudaFreeHost(ctx->h_mel);
            break;
        case MEM_MANAGED:
            cudaFree(ctx->d_audio);      // managed pointers freed with cudaFree
            cudaFree(ctx->d_mel);
            break;
    }
    cufftDestroy(ctx->plan);
    delete ctx;
}

// One-shot convenience wrapper.
void compute_log_mel_cuda(
    const float*     audio,
    size_t           n_samples,
    const float*     hann_window,
    const float*     mel_filterbank,
    const MelConfig& cfg,
    float*           log_mel
) {
    MelContext* ctx = mel_context_create(n_samples, hann_window, mel_filterbank, cfg);
    mel_context_run(ctx, audio, log_mel);
    mel_context_destroy(ctx);
}
