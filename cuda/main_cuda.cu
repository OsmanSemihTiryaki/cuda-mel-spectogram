// CUDA naive driver. Mirrors the CPU reference driver so the two are directly
// comparable. Reads the same binary inputs, writes a float32 output, times the
// pipeline using the persistent-context API, and optionally validates against a
// reference output file.
//
// Usage:
//   ./mel_cuda audio.bin hann.bin filterbank.bin output.bin
//       [--runs N] [--ref ref.bin]
//       [--naive | --tiled | --tile T]   (mel-multiply kernel; default --tile 16)
//       [--mem malloc|mapped|managed]    (buffer placement; default malloc)
//
// All input/output files are raw little-endian float32, no header.

#include "mel_pipeline_cuda.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

#include <cuda_runtime.h>

static std::vector<float> read_floats(const char* path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        std::fprintf(stderr, "ERROR: cannot open %s\n", path);
        std::exit(EXIT_FAILURE);
    }
    std::streamsize n_bytes = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<float> data(n_bytes / sizeof(float));
    f.read(reinterpret_cast<char*>(data.data()), n_bytes);
    return data;
}

static void write_floats(const char* path, const float* data, size_t n) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(data), n * sizeof(float));
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(stderr,
            "Usage: %s audio.bin hann.bin filterbank.bin output.bin "
            "[--runs N] [--ref ref.bin] [--naive|--tiled|--tile T] "
            "[--mem malloc|mapped|managed]\n", argv[0]);
        return EXIT_FAILURE;
    }

    int n_runs = 100;
    const char* ref_path  = nullptr;
    bool use_tiled = true;   // default to the optimized path
    int tile_size = 16;
    MemMode mem_mode  = MEM_MALLOC;
    for (int i = 5; i < argc; ++i) {
        if (std::strcmp(argv[i], "--runs") == 0 && i + 1 < argc) {
            n_runs = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--ref") == 0 && i + 1 < argc) {
            ref_path = argv[++i];
        } else if (std::strcmp(argv[i], "--naive") == 0) {
            use_tiled = false;
        } else if (std::strcmp(argv[i], "--tiled") == 0) {
            use_tiled = true;
        } else if (std::strcmp(argv[i], "--tile") == 0 && i + 1 < argc) {
            tile_size = std::atoi(argv[++i]);
            use_tiled = true;
        } else if (std::strcmp(argv[i], "--mem") == 0 && i + 1 < argc) {
            const char* m = argv[++i];
            if (std::strcmp(m, "malloc")  == 0) mem_mode = MEM_MALLOC;
            else if (std::strcmp(m, "mapped")  == 0) mem_mode = MEM_MAPPED;
            else if (std::strcmp(m, "managed") == 0) mem_mode = MEM_MANAGED;
            else { std::fprintf(stderr, "unknown --mem %s (malloc|mapped|managed)\n", m);
                   return EXIT_FAILURE; }
        }
    }

    const auto audio      = read_floats(argv[1]);
    const auto hann       = read_floats(argv[2]);
    const auto filterbank = read_floats(argv[3]);

    MelConfig cfg{};
    cfg.sample_rate = 16000;
    cfg.frame_length = 400;
    cfg.hop_length = 160;
    cfg.fft_size = 512;
    cfg.n_mels = 128;
    cfg.log_eps = 1e-10f;

    const int n_bins = cfg.fft_size / 2 + 1;
    if ((int)hann.size() != cfg.frame_length) {
        std::fprintf(stderr, "ERROR: hann size %zu != frame_length %d\n",
                     hann.size(), cfg.frame_length);
        return EXIT_FAILURE;
    }
    if ((int)filterbank.size() != n_bins * cfg.n_mels) {
        std::fprintf(stderr, "ERROR: filterbank size %zu != %d\n",
                     filterbank.size(), n_bins * cfg.n_mels);
        return EXIT_FAILURE;
    }

    // n_frames via the same formula the library uses.
    size_t n_frames = 0;
    if (audio.size() >= (size_t)cfg.frame_length) {
        n_frames = (audio.size() - cfg.frame_length) / cfg.hop_length + 1;
    }
    if (n_frames == 0) {
        std::fprintf(stderr, "ERROR: audio shorter than one frame\n");
        return EXIT_FAILURE;
    }
    std::vector<float> log_mel(n_frames * cfg.n_mels);

    // Build context once (allocations + cuFFT plan + static copies).
    // If zero-copy mapped memory is requested, the device must allow host
    // mapping. This flag has to be set before the CUDA context is created,
    // i.e. before the first CUDA call (mel_context_create below).
    if (mem_mode == MEM_MAPPED) {
        cudaError_t e = cudaSetDeviceFlags(cudaDeviceMapHost);
        if (e != cudaSuccess) {
            std::fprintf(stderr, "cudaSetDeviceFlags failed: %s\n",
                         cudaGetErrorString(e));
            return EXIT_FAILURE;
        }
    }

    MelContext* ctx = mel_context_create(audio.size(), hann.data(),
                                         filterbank.data(), cfg, mem_mode);
    mel_context_set_kernel(ctx, use_tiled, tile_size);

    // Warm up (also triggers lazy CUDA context init / JIT).
    for (int i = 0; i < 5; ++i) {
        mel_context_run(ctx, audio.data(), log_mel.data());
    }

    // Timed runs using CUDA events (device-side timing of the full run() call,
    // including H2D and D2H copies).
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float total_ms = 0.0f;
    for (int i = 0; i < n_runs; ++i) {
        cudaEventRecord(start);
        mel_context_run(ctx, audio.data(), log_mel.data());
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms;
    }

    std::printf("\nCUDA mel benchmark\n");
    if (use_tiled) {
        std::printf("  kernel        : tiled+fused (TILE=%d)\n", tile_size);
    } else {
        std::printf("  kernel        : naive (separate power + naive GEMM)\n");
    }
    const char* mem_name = (mem_mode == MEM_MALLOC) ? "malloc+memcpy"
                         : (mem_mode == MEM_MAPPED) ? "mapped (zero-copy)"
                         : "managed (unified)";
    std::printf("  memory        : %s\n", mem_name);
    std::printf("  audio samples : %zu (%.3f s @ %d Hz)\n",
                audio.size(), (double)audio.size() / cfg.sample_rate,
                cfg.sample_rate);
    std::printf("  output shape  : (%zu, %d)\n", n_frames, cfg.n_mels);
    std::printf("  warmup runs   : 5\n");
    std::printf("  timed runs    : %d\n", n_runs);
    std::printf("  --- timings (ms, full run incl. H2D+D2H) ---\n");
    std::printf("  mean per run  : %.3f\n", total_ms / n_runs);
    std::printf("  total elapsed : %.3f\n", total_ms);

    write_floats(argv[4], log_mel.data(), log_mel.size());
    std::printf("Output written to %s (%zu floats)\n", argv[4], log_mel.size());

    // Per-stage breakdown, averaged over the same number of runs.
    {
        MelTimings acc{};
        for (int i = 0; i < n_runs; ++i) {
            MelTimings t{};
            mel_context_run_profiled(ctx, audio.data(), log_mel.data(), &t);
            acc.h2d      += t.h2d;
            acc.k1_frame += t.k1_frame;
            acc.fft      += t.fft;
            acc.k2_power += t.k2_power;
            acc.k3_mel   += t.k3_mel;
            acc.k4_log   += t.k4_log;
            acc.d2h      += t.d2h;
            acc.total    += t.total;
        }
        float inv = 1.0f / n_runs;
        float tot = acc.total * inv;
        auto pct = [&](float v) { return 100.0f * v / acc.total; };

        std::printf("\n  --- per-stage breakdown (mean over %d runs) ---\n", n_runs);
        std::printf("  %-14s %8.4f ms  (%5.1f%%)\n", "H2D copy",   acc.h2d * inv,      pct(acc.h2d));
        std::printf("  %-14s %8.4f ms  (%5.1f%%)\n", "K1 frame",   acc.k1_frame * inv, pct(acc.k1_frame));
        std::printf("  %-14s %8.4f ms  (%5.1f%%)\n", "cuFFT",      acc.fft * inv,      pct(acc.fft));
        // The mel-multiply stage is timed as one segment (held in k2_power):
        // naive mode = power kernel + naive GEMM; tiled mode = fused kernel.
        const char* mel_label = use_tiled ? "mel (fused)" : "mel (pow+GEMM)";
        std::printf("  %-14s %8.4f ms  (%5.1f%%)\n", mel_label,    acc.k2_power * inv, pct(acc.k2_power));
        std::printf("  %-14s %8.4f ms  (%5.1f%%)\n", "K4 log",     acc.k4_log * inv,   pct(acc.k4_log));
        std::printf("  %-14s %8.4f ms  (%5.1f%%)\n", "D2H copy",   acc.d2h * inv,      pct(acc.d2h));
        std::printf("  %-14s %8.4f ms\n", "sum", tot);
    }

    // Optional validation against a reference output (e.g. the CPU result).
    if (ref_path) {
        const auto ref = read_floats(ref_path);
        if (ref.size() != log_mel.size()) {
            std::fprintf(stderr,
                "WARN: ref size %zu != output size %zu, skipping compare\n",
                ref.size(), log_mel.size());
        } else {
            float max_abs = 0.0f;
            double sum_abs = 0.0;
            for (size_t i = 0; i < ref.size(); ++i) {
                float d = std::fabs(log_mel[i] - ref[i]);
                if (d > max_abs) max_abs = d;
                sum_abs += d;
            }
            std::printf("  --- validation vs %s ---\n", ref_path);
            std::printf("  max abs error : %.6e\n", max_abs);
            std::printf("  mean abs error: %.6e\n", sum_abs / ref.size());
            std::printf("  %s\n", max_abs < 1e-3f ? "PASS" : "FAIL");
        }
    }

    mel_context_destroy(ctx);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return EXIT_SUCCESS;
}
