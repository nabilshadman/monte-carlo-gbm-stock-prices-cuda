/*
 monte_carlo_gbm.cu

 Purpose:
   Monte Carlo simulation of stock prices following Geometric Brownian Motion (GBM)
   S(t+dt) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z), Z ~ N(0,1)

 Target:
   NVIDIA A100 GPU (Ampere, sm_80). Uses double precision for finance-grade accuracy.

 Build:
   nvcc monte_carlo_gbm.cu -o monte_carlo_gbm -arch=sm_80

 Example run:
   ./monte_carlo_gbm 10000000 252 100.0 0.05 0.2 1.0
   -> Simulate 10,000,000 paths, 252 steps (daily for 1 year), S0=100, mu=5%, sigma=20%, T=1 year.

 Notes / Best practices:
  - Uses curandStatePhilox4_32_10_t RNG (Philox) with per-thread sequences for reproducibility.
  - Each GPU thread simulates exactly one path (vectorize by number of paths).
  - Use -arch=sm_80 for A100; for other devices change accordingly.
  - Results are copied back to host for final statistics (mean, std). For very large path counts
    consider on-GPU reductions to avoid host memory pressure.
  - For production: consider batching, multi-GPU, and fused reductions to reduce host-device traffic.
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <vector>
#include <iostream>
#include <iomanip>

#include <cuda.h>
#include <curand_kernel.h>

// ----------------------- Error checking macros ------------------------------
#define CUDA_CALL(expr)                                               \
    do {                                                               \
        cudaError_t err = (expr);                                      \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));      \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

#define CURAND_CALL(expr)                                              \
    do {                                                               \
        curandStatus_t st = (expr);                                    \
        if (st != CURAND_STATUS_SUCCESS) {                             \
            fprintf(stderr, "cuRAND error %s:%d: %d\n",                \
                    __FILE__, __LINE__, (int)st);                      \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

// ----------------------- GPU kernel ----------------------------------------
// Each thread simulates one independent GBM path and writes final price to d_out[idx].
__global__ void gbm_simulate_kernel_double(
    double *d_out,                 // output final price per path
    const uint64_t n_paths,        // total number of paths
    const int n_steps,             // number of time steps per path
    const double S0,               // initial asset price
    const double mu,               // drift
    const double sigma,            // volatility
    const double dt,               // timestep (T / n_steps)
    const unsigned long long seed  // RNG seed
) {
    const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_paths) return;

    // Initialize Philox RNG state for reproducible parallel streams
    curandStatePhilox4_32_10_t state;
    curand_init(seed, /* sequence */ idx, /* offset */ 0ULL, &state);

    double s = S0;
    const double sd_sqrt_dt = sigma * sqrt(dt);
    const double drift_term = (mu - 0.5 * sigma * sigma) * dt;

    // Simulate path
    for (int step = 0; step < n_steps; ++step) {
        // Pull standard normal from curand (double precision)
        double z = curand_normal_double(&state);
        // GBM update
        s *= exp(drift_term + sd_sqrt_dt * z);
    }

    d_out[idx] = s;
}

// ----------------------- Host helpers --------------------------------------
void usage_and_exit(const char *prog) {
    fprintf(stderr,
        "Usage: %s <n_paths> <n_steps> <S0> <mu> <sigma> <T_years>\n"
        " Example: %s 10000000 252 100.0 0.05 0.2 1.0\n",
        prog, prog);
    exit(EXIT_FAILURE);
}

// Compute mean and standard deviation on host
void compute_stats_host(const std::vector<double> &v, double &mean, double &stddev) {
    const size_t n = v.size();
    if (n == 0) { mean = stddev = NAN; return; }
    long double sum = 0.0L;
    for (double x : v) sum += (long double)x;
    mean = (double)(sum / (long double)n);

    long double ssum = 0.0L;
    for (double x : v) {
        long double d = (long double)x - (long double)mean;
        ssum += d * d;
    }
    stddev = (double)sqrt((double)(ssum / (long double)n));
}

// ----------------------- Main ------------------------------------------------
int main(int argc, char *argv[]) {
    if (argc != 7) usage_and_exit(argv[0]);

    // Parse CLI
    const uint64_t n_paths = strtoull(argv[1], nullptr, 10);
    const int n_steps = atoi(argv[2]);
    const double S0 = atof(argv[3]);
    const double mu = atof(argv[4]);
    const double sigma = atof(argv[5]);
    const double T = atof(argv[6]);

    if (n_paths == 0 || n_steps <= 0 || S0 <= 0.0 || T <= 0.0) {
        fprintf(stderr, "Invalid numeric arguments.\n");
        usage_and_exit(argv[0]);
    }

    const double dt = T / (double)n_steps;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Monte Carlo GBM settings:\n"
              << "  Paths       : " << n_paths << "\n"
              << "  Steps/path  : " << n_steps << "\n"
              << "  S0          : " << S0 << "\n"
              << "  mu          : " << mu << "\n"
              << "  sigma       : " << sigma << "\n"
              << "  T (years)   : " << T << "\n"
              << "  dt          : " << dt << "\n";

    // Device allocation
    double *d_out = nullptr;
    size_t out_bytes = sizeof(double) * (size_t)n_paths;
    CUDA_CALL(cudaMalloc((void**)&d_out, out_bytes));

    // Kernel launch configuration
    const int threads_per_block = 256;
    const uint64_t blocks = (n_paths + (uint64_t)threads_per_block - 1ULL) / (uint64_t)threads_per_block;

    // RNG seed - use steady_clock time for variability; can be parameterized
    const unsigned long long seed = static_cast<unsigned long long>(
        std::chrono::steady_clock::now().time_since_epoch().count());

    // GPU timing with events
    cudaEvent_t start, stop;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));
    CUDA_CALL(cudaEventRecord(start));

    // Launch kernel
    gbm_simulate_kernel_double<<<(int)blocks, threads_per_block>>>(
        d_out, n_paths, n_steps, S0, mu, sigma, dt, seed
    );

    // Check kernel launch
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaEventRecord(stop));
    CUDA_CALL(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CALL(cudaEventElapsedTime(&ms, start, stop));

    // Copy results back to host
    std::vector<double> h_out;
    try {
        h_out.resize((size_t)n_paths);
    } catch (const std::bad_alloc &) {
        fprintf(stderr, "Host allocation failed for %zu bytes\n", out_bytes);
        CUDA_CALL(cudaFree(d_out));
        exit(EXIT_FAILURE);
    }
    CUDA_CALL(cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));

    // Compute statistics
    double mean = 0.0, stddev = 0.0;
    compute_stats_host(h_out, mean, stddev);

    // Print summary
    std::cout << "GPU kernel time (ms): " << ms << " ms\n";
    std::cout << "Results (final price per path):\n";
    std::cout << "  Mean final price : " << mean << "\n";
    std::cout << "  StdDev final price: " << stddev << "\n";

    // Print first few samples for sanity check
    const size_t show = (n_paths < 10) ? n_paths : 10;
    std::cout << "  First " << show << " simulated final prices:\n";
    for (size_t i = 0; i < show; ++i) {
        std::cout << "    [" << i << "] " << h_out[i] << "\n";
    }

    // Clean up
    CUDA_CALL(cudaFree(d_out));
    CUDA_CALL(cudaEventDestroy(start));
    CUDA_CALL(cudaEventDestroy(stop));

    return 0;
}

