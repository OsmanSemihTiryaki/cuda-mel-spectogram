# Build the naive CUDA mel pipeline.
# Target: Jetson Orin NX (Ampere, compute capability 8.7).
#
# Requires the CUDA toolkit (nvcc) and cuFFT, both included in JetPack.
#
# Targets:
#   make - build mel_cuda
#   make clean  - remove build artifacts
#
# If building on a different GPU, override ARCH, e.g.:
#   make ARCH=sm_89        (RTX 4070, Ada)

NVCC   ?= nvcc
ARCH   ?= sm_87
NVFLAGS = -O3 -std=c++17 -arch=$(ARCH) -lineinfo
LDLIBS  = -lcufft

TARGET  = mel_cuda
SRCS    = main_cuda.cu mel_pipeline_cuda.cu
HDRS    = mel_pipeline_cuda.h mel_config.h

$(TARGET): $(SRCS) $(HDRS)
	$(NVCC) $(NVFLAGS) $(SRCS) -o $@ $(LDLIBS)

clean:
	rm -f $(TARGET)

.PHONY: clean
