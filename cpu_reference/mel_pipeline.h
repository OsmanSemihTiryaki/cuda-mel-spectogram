// CPU reference implementation of the AST mel-spectrogram pipeline.
// Single-threaded, plain C++17, FFT via FFTW (single-precision).
//
// Purpose:
//   1. Numerical ground truth for validating GPU output (max-abs-error < 1e-3).
//   2. Algorithmic baseline showing the same pipeline without parallelism.
//
// Pipeline:
//   audio -> [frame + Hann window + zero-pad] -> [FFT] -> [power]
//         -> [mel filterbank multiply] -> [log10] -> log-mel spectrogram
//
// Layout convention: every multi-dim array is row-major.

#pragma once

#include <cstddef>

struct MelConfig {
    int sample_rate; // 16000
    int frame_length; // 400  (25 ms at 16 kHz)
    int hop_length; // 160  (10 ms at 16 kHz)
    int fft_size; // 512  (next power of 2 >= frame_length)
    int n_mels; // 128
    float log_eps; // 1e-10  (clamp before log to avoid -inf)
};

// Number of output frames produced from `n_samples` input samples.
// Returns 0 if the audio is shorter than one full frame.
size_t compute_n_frames(size_t n_samples, int frame_length, int hop_length);

// Compute the log-mel spectrogram.
//
// Inputs:
//   audio : float32 PCM, length n_samples, range [-1, 1]
//   hann_window : precomputed Hann window, length cfg.frame_length
//   mel_filterbank : row-major [cfg.fft_size/2 + 1, cfg.n_mels]
//
// Output:
//   log_mel : row-major [n_frames, cfg.n_mels]
//             Caller must preallocate. n_frames = compute_n_frames(...).
void compute_log_mel(
    const float* audio,
    size_t n_samples,
    const float* hann_window,
    const float* mel_filterbank,
    const MelConfig& cfg,
    float* log_mel
);
