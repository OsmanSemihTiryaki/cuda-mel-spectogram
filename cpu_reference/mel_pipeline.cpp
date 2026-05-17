#include "mel_pipeline.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include <fftw3.h>

size_t compute_n_frames(size_t n_samples, int frame_length, int hop_length) {
    if (n_samples < static_cast<size_t>(frame_length)) {
        return 0;
    }
    return (n_samples - static_cast<size_t>(frame_length)) /
           static_cast<size_t>(hop_length) + 1;
}

void compute_log_mel(
    const float* audio,
    size_t n_samples,
    const float* hann_window,
    const float* mel_filterbank,
    const MelConfig& cfg,
    float* log_mel
) {
    const int N = cfg.fft_size;
    const int n_bins = N / 2 + 1;
    const size_t n_frames = compute_n_frames(n_samples, cfg.frame_length,
                                             cfg.hop_length);

    // FFTW single-precision real-to-complex 1D plan, reused across frames.
    float* fft_in  = fftwf_alloc_real(static_cast<size_t>(N));
    fftwf_complex* fft_out = fftwf_alloc_complex(static_cast<size_t>(n_bins));
    fftwf_plan plan = fftwf_plan_dft_r2c_1d(N, fft_in, fft_out, FFTW_ESTIMATE);

    std::vector<float> power(static_cast<size_t>(n_bins));

    for (size_t f = 0; f < n_frames; ++f) {
        const size_t start = f * static_cast<size_t>(cfg.hop_length);

        // Stage 1: frame + window. Zero-pad up to fft_size since
        // frame_length (400) < fft_size (512).
        for (int n = 0; n < cfg.frame_length; ++n) {
            fft_in[n] = audio[start + static_cast<size_t>(n)] * hann_window[n];
        }
        for (int n = cfg.frame_length; n < N; ++n) {
            fft_in[n] = 0.0f;
        }

        // Stage 2: FFT.
        fftwf_execute(plan);

        // Stage 3: power spectrum.
        for (int k = 0; k < n_bins; ++k) {
            const float re = fft_out[k][0];
            const float im = fft_out[k][1];
            power[static_cast<size_t>(k)] = re * re + im * im;
        }

        // Stage 4: mel filterbank multiply.
        // mel_filterbank is row-major [n_bins, n_mels].
        // The naive triple loop here is the algorithmic ancestor of the GPU
        // tiled GEMM kernel — this is the reference output the GPU must match.
        float* out_row = log_mel + f * static_cast<size_t>(cfg.n_mels);
        for (int m = 0; m < cfg.n_mels; ++m) {
            float sum = 0.0f;
            for (int k = 0; k < n_bins; ++k) {
                sum += power[static_cast<size_t>(k)] *
                       mel_filterbank[static_cast<size_t>(k) *
                                      static_cast<size_t>(cfg.n_mels) +
                                      static_cast<size_t>(m)];
            }
            // Stage 5: log compression.
            out_row[m] = 10.0f * std::log10(std::max(sum, cfg.log_eps));
        }
    }

    fftwf_destroy_plan(plan);
    fftwf_free(fft_in);
    fftwf_free(fft_out);
}
