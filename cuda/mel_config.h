// Shared configuration for the mel-spectrogram pipeline.
// Included by both the CPU reference and the CUDA implementation so the two
// agree byte-for-byte on parameters.

#pragma once

struct MelConfig {
    int   sample_rate;    // 16000
    int   frame_length;   // 400  (25 ms at 16 kHz)
    int   hop_length;     // 160  (10 ms at 16 kHz)
    int   fft_size;       // 512  (next power of 2 >= frame_length)
    int   n_mels;         // 128
    float log_eps;        // 1e-10  (clamp before log to avoid -inf)
};
