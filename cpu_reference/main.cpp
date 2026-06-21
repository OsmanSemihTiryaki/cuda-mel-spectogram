// CPU reference driver.
//
// Reads pre-generated binary files and produces a log-mel spectrogram.
// Times the pipeline so it can be compared to the HF NumPy baseline and to
// the GPU implementation.
//
// Usage:
//   ./cpu_reference audio.bin hann.bin filterbank.bin output.bin [--runs N]
//

#include "mel_pipeline.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static std::vector<float> read_floats(const char* path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        std::fprintf(stderr, "ERROR: cannot open %s\n", path);
        std::exit(EXIT_FAILURE);
    }
    const std::streamsize n_bytes = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<float> data(static_cast<size_t>(n_bytes) / sizeof(float));
    f.read(reinterpret_cast<char*>(data.data()), n_bytes);
    return data;
}

static void write_floats(const char* path, const float* data, size_t n) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(data),
            static_cast<std::streamsize>(n * sizeof(float)));
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(stderr,
            "Usage: %s audio.bin hann.bin filterbank.bin output.bin [--runs N]\n",
            argv[0]);
        return EXIT_FAILURE;
    }

    int n_runs = 100;
    for (int i = 5; i < argc; ++i) {
        if (std::strcmp(argv[i], "--runs") == 0 && i + 1 < argc) {
            n_runs = std::atoi(argv[++i]);
        }
    }

    const auto audio = read_floats(argv[1]);
    const auto hann = read_floats(argv[2]);
    const auto filterbank = read_floats(argv[3]);

    MelConfig cfg{};
    cfg.sample_rate = 16000;
    cfg.frame_length = 400;
    cfg.hop_length = 160;
    cfg.fft_size = 512;
    cfg.n_mels = 128;
    cfg.log_eps = 1e-10f;

    if (static_cast<int>(hann.size()) != cfg.frame_length) {
        std::fprintf(stderr, "ERROR: hann size %zu != frame_length %d\n",
                     hann.size(), cfg.frame_length);
        return EXIT_FAILURE;
    }
    const int n_bins = cfg.fft_size / 2 + 1;
    if (static_cast<int>(filterbank.size()) != n_bins * cfg.n_mels) {
        std::fprintf(stderr,
            "ERROR: filterbank size %zu != n_bins * n_mels = %d\n",
            filterbank.size(), n_bins * cfg.n_mels);
        return EXIT_FAILURE;
    }

    const size_t n_frames = compute_n_frames(audio.size(), cfg.frame_length,
                                             cfg.hop_length);
    if (n_frames == 0) {
        std::fprintf(stderr, "ERROR: audio shorter than one frame\n");
        return EXIT_FAILURE;
    }
    std::vector<float> log_mel(n_frames * static_cast<size_t>(cfg.n_mels));

    // Warm up.
    for (int i = 0; i < 5; ++i) {
        compute_log_mel(audio.data(), audio.size(),
                        hann.data(), filterbank.data(),
                        cfg, log_mel.data());
    }

    // Timed runs.
    const auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_runs; ++i) {
        compute_log_mel(audio.data(), audio.size(),
                        hann.data(), filterbank.data(),
                        cfg, log_mel.data());
    }
    const auto t1 = std::chrono::high_resolution_clock::now();
    const double total_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::printf("\nCPU reference benchmark\n");
    std::printf("  audio samples : %zu (%.3f s @ %d Hz)\n",
                audio.size(),
                static_cast<double>(audio.size()) / cfg.sample_rate,
                cfg.sample_rate);
    std::printf("  output shape  : (%zu, %d)\n", n_frames, cfg.n_mels);
    std::printf("  warmup runs   : 5\n");
    std::printf("  timed runs    : %d\n", n_runs);
    std::printf("  --- timings (ms) ---\n");
    std::printf("  mean per run  : %.3f\n", total_ms / n_runs);
    std::printf("  total elapsed : %.3f\n", total_ms);

    write_floats(argv[4], log_mel.data(), log_mel.size());
    std::printf("Output written to %s (%zu floats, %zu bytes)\n",
                argv[4], log_mel.size(), log_mel.size() * sizeof(float));

    return EXIT_SUCCESS;
}
