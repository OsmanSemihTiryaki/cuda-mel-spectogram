# Build the single-threaded CPU reference.
# Requires FFTW (single-precision):
#   sudo apt install libfftw3-dev
#
# Targets:
#   make           - build cpu_reference
#   make clean     - remove build artifacts

CXX      ?= g++
CXXFLAGS  = -O3 -std=c++17 -Wall -Wextra -Wpedantic
LDLIBS    = -lfftw3f -lm

TARGET    = cpu_reference
SRCS      = main.cpp mel_pipeline.cpp
HDRS      = mel_pipeline.h

$(TARGET): $(SRCS) $(HDRS)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $@ $(LDLIBS)

clean:
	rm -f $(TARGET)

.PHONY: clean
