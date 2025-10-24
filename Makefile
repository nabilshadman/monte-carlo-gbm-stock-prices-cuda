# Makefile for Monte Carlo GBM CUDA Project

# Compiler
NVCC = nvcc

# Target executable
TARGET = monte_carlo_gbm

# Source files
SRC = monte_carlo_gbm.cu

# Compiler flags
NVCC_FLAGS = -arch=sm_80
NVCC_FLAGS_DEBUG = -g -G -arch=sm_80

# Default target
all: $(TARGET)

# Build release version
$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $(SRC) -o $(TARGET)
	@echo "Build complete: $(TARGET)"

# Build debug version
debug: $(SRC)
	$(NVCC) $(NVCC_FLAGS_DEBUG) $(SRC) -o $(TARGET)_debug
	@echo "Debug build complete: $(TARGET)_debug"

# Run example
run: $(TARGET)
	./$(TARGET) 10000000 252 100.0 0.05 0.2 1.0

# Profile with Nsight Systems
profile: $(TARGET)
	nsys profile --trace=cuda,nvtx --stats=true --output=$(TARGET)_profile --force-overwrite=true \
		./$(TARGET) 10000000 252 100.0 0.05 0.2 1.0

# View profile statistics
profile-stats:
	nsys stats --report cuda_gpu_kern_sum $(TARGET)_profile.nsys-rep

# Clean build artifacts
clean:
	rm -f $(TARGET) $(TARGET)_debug
	@echo "Cleaned build artifacts"

# Clean everything including profile data
cleanall: clean
	rm -f *.nsys-rep *.qdrep *.sqlite
	@echo "Cleaned all files"

# Help
help:
	@echo "Monte Carlo GBM Makefile"
	@echo "========================"
	@echo "Targets:"
	@echo "  all           - Build release version (default)"
	@echo "  debug         - Build debug version with device debug info"
	@echo "  run           - Build and run with example parameters"
	@echo "  profile       - Build and profile with Nsight Systems"
	@echo "  profile-stats - View kernel statistics from profile"
	@echo "  clean         - Remove build artifacts"
	@echo "  cleanall      - Remove all build and profile files"
	@echo "  help          - Show this help message"
	@echo ""
	@echo "Example usage:"
	@echo "  make              # Build release version"
	@echo "  make run          # Build and run"
	@echo "  make profile      # Profile the application"
	@echo "  make clean        # Clean up"

.PHONY: all debug run profile profile-stats clean cleanall help