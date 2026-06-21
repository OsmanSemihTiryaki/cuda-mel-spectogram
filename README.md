# cuda-mel-spectogram

GPU-accelerated mel spectrogram computation in CUDA for the Audio Spectrogram
Transformer (AST) preprocessing pipeline, benchmarked on a Jetson Orin NX.

The HuggingFace `ASTFeatureExtractor` computes mel spectrograms on the CPU,
leaving the GPU idle during preprocessing. This project moves the full pipeline
(framing + Hann window → FFT → power → mel filterbank multiply → log) onto the
GPU. The mel multiply has a naive baseline and a tiled + fused implementation,
and the audio/output buffers support three memory-placement strategies for the
Jetson's unified memory.

**Result:** ~25–59× faster end-to-end than a single-threaded C++ reference;
36% of the remaining time eliminated by using unified memory instead of an
explicit copy.

## Layout

```
cpu_reference/    single-threaded C++17 reference (FFTW), the numerical ground truth
cuda/             the CUDA implementation (both kernels, all memory modes) + sweeps
helper_scripts/   filterbank/Hann precompute, HF baseline, sweep-input generation
```

## Requirements

- CUDA toolkit (JetPack on Jetson) with cuFFT, `nvcc` on `PATH`
- `libfftw3-dev` (for the CPU reference)
- Python 3.10 with `numpy<2`, `librosa`, `transformers`, `torch` (CPU), `matplotlib`

```bash
sudo apt install -y libfftw3-dev python3-venv
python3 -m venv ~/mel_venv && source ~/mel_venv/bin/activate
pip install "numpy<2" librosa transformers soundfile matplotlib
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Build & run

```bash
# 1. Generate the static inputs (filterbank + Hann window) and test audio
cd helper_scripts
python precompute_filterbank.py --filterbank-out ../cpu_reference/filterbank.bin \
                                --hann-out ../cpu_reference/hann.bin
python hf_benchmark.py --window 1.0 --runs 200 --save-audio ../cpu_reference/audio.bin

# 2. CPU reference -> produces output.bin (the ground truth)
cd ../cpu_reference && make
./cpu_reference audio.bin hann.bin filterbank.bin output.bin --runs 100

# 3. CUDA pipeline, validated against the CPU output
cd ../cuda && make
./mel_cuda ../cpu_reference/audio.bin ../cpu_reference/hann.bin \
           ../cpu_reference/filterbank.bin cuda_out.bin \
           --runs 100 --ref ../cpu_reference/output.bin
```

`mel_cuda` flags: `--naive | --tiled | --tile T` (mel kernel, default `--tile 16`),
`--mem malloc|mapped|managed` (buffer placement), `--runs N`, `--ref ref.bin`.

## Benchmark sweeps

```bash
cd cuda
python ../helper_scripts/gen_sweep_inputs.py --out-dir sweep_data --lengths 0.5 1 2 5 10
python run_sweeps.py --mel-cuda ./mel_cuda --cpu-ref ../cpu_reference/cpu_reference \
                     --data-dir sweep_data --lengths 0.5 1 2 5 10 --runs 200
python plot_sweeps.py --length sweep_length.csv --tile sweep_tile.csv --mem sweep_mem.csv
```

This runs the audio-length, naive-vs-tiled, tile-size, and memory-placement
sweeps and writes the result figures.

## Configuration

16 kHz, frame 400, hop 160, FFT 512, 128 mel bins, Hann window, librosa slaney
mel scale. One second of audio → 98 × 128 log-mel output. Target hardware:
Jetson Orin NX (Ampere, `sm_87`).
