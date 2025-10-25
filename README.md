# GPU-Accelerated Monte Carlo Simulation of Stock Prices using Geometric Brownian Motion (CUDA)

## Overview

This project implements a **Monte Carlo simulation** of **Geometric Brownian Motion (GBM)** to model stochastic stock price dynamics. The entire computation runs on the **GPU using NVIDIA CUDA**, enabling the simulation of millions of independent price paths in parallel with high performance.

The simulation's goal is to efficiently estimate statistical properties of terminal stock prices under GBM dynamics — a foundational process in quantitative finance and computational stochastic modeling.

### Tech Stack
- **C++17** - Host code and application logic
- **CUDA** - GPU kernel implementation and parallel computing
- **cuRAND** - On-device random number generation
- **Make** - Build automation
- **Nsight Systems** - Performance profiling and analysis

---

## 1. Theoretical Background

### Geometric Brownian Motion (GBM)
In continuous time, the stock price $S_t$ follows the stochastic differential equation:

$$dS_t = \mu S_t dt + \sigma S_t dW_t$$

where:
- $\mu$ = expected rate of return (drift),
- $\sigma$ = volatility,
- $W_t$ = standard Brownian motion.

The analytical solution is:

$$S_t = S_0 \exp \left( \left( \mu - \frac{1}{2}\sigma^2 \right)t + \sigma W_t \right)$$

Monte Carlo simulation discretizes this process and evolves prices over $N$ time steps for each of $M$ simulated paths.

### Expected Results

For a time horizon $T$:

$$E[S_T] = S_0 e^{\mu T}, \quad Var(S_T) = S_0^2 e^{2\mu T}(e^{\sigma^2 T} - 1)$$

These analytical benchmarks are used to validate the numerical simulation.

---

## 2. Implementation Architecture

### GPU Design

Each **CUDA thread** simulates **one independent price path**:
1. Initializes with $S_0$
2. Iteratively updates over $N$ time steps using random Gaussian draws.
3. Writes the final price $S_T$ to global memory.

### Algorithmic Steps

1. **Random Number Generation**  
   Uses NVIDIA's **cuRAND** library to produce standard normal variates efficiently on-device.

2. **Parallel Path Simulation**  
   Each thread executes the GBM update rule:

   $$S_{t+\Delta t} = S_t \times \exp\left((\mu - \frac{1}{2}\sigma^2)\Delta t + \sigma \sqrt{\Delta t} Z_t\right)$$
   
   where $Z_t \sim \mathcal{N}(0,1)$.

3. **Reduction and Statistics**  
   Final prices are transferred back to the CPU for statistical post-processing (mean, standard deviation, etc.).

---

## 3. Code Structure

```
monte_carlo_gbm_gpu/
│
├── monte_carlo_gbm.cu        # Main CUDA source file
├── Makefile                  # Build automation
├── README.md                 # Project documentation (this file)
├── sample_run_1_a100.txt     # Sample run output (configuration 1)
├── sample_run_2_a100.txt     # Sample run output (configuration 2)
└── sample_run_3_a100.txt     # Sample run output (configuration 3)
```

### Key Components in Code
| Section | Purpose |
|----------|----------|
| `curand_init()` | Initializes the cuRAND generator |
| `gbm_simulate_kernel_double()` | Core GPU kernel — each thread simulates one path |
| `compute_stats_host()` | Computes mean and standard deviation of final prices |
| `main()` | Parses arguments, allocates memory, launches GPU kernel |

---

## 4. Compilation and Execution

### Requirements
- **NVIDIA GPU** with Compute Capability ≥ 8.0 (e.g., A100)
- **CUDA Toolkit ≥ 12.0**
- **C++17 or later**

### Compilation

#### Option 1: Using Make (Recommended)

```bash
# Build the project
make

# Build and run with example parameters
make run

# Build debug version
make debug

# View all available targets
make help
```

#### Option 2: Manual Compilation

```bash
nvcc monte_carlo_gbm.cu -o monte_carlo_gbm -arch=sm_80
```

For portability across GPU architectures:
```bash
nvcc monte_carlo_gbm.cu -o monte_carlo_gbm -gencode arch=compute_80,code=sm_80
```

### Execution

Run the compiled executable with the following parameters:

```bash
./monte_carlo_gbm <n_paths> <n_steps> <S0> <mu> <sigma> <T_years>
```

**Example:**
```bash
./monte_carlo_gbm 10000000 252 100.0 0.05 0.2 1.0
```

**Parameters:**
- `n_paths`: Number of Monte Carlo simulation paths (e.g., 10000000)
- `n_steps`: Number of time steps per path (e.g., 252 for daily trading days in a year)
- `S0`: Initial stock price (e.g., 100.0)
- `mu`: Expected rate of return / drift (e.g., 0.05 for 5%)
- `sigma`: Volatility (e.g., 0.2 for 20%)
- `T_years`: Time horizon in years (e.g., 1.0)

---

## 5. Sample Output (10 Million Paths, 252 Steps)

```
Monte Carlo GBM settings:
  Paths       : 10000000
  Steps/path  : 252
  S0          : 100.000000
  mu          : 0.050000
  sigma       : 0.200000
  T (years)   : 1.000000
  dt          : 0.003968
GPU kernel time (ms): 40.006657 ms
Results (final price per path):
  Mean final price : 105.139082
  StdDev final price: 21.248964
  First 10 simulated final prices:
    [0] 92.736818
    [1] 121.266588
    [2] 78.763418
    [3] 95.508716
    [4] 130.206386
    [5] 82.899322
    [6] 112.220961
    [7] 127.625210
    [8] 101.675693
    [9] 169.514530
```

---

## 6. Hardware Environment (Benchmark System)

| Component | Specification |
|------------|----------------|
| **CPU** | AMD EPYC 7713 (64 cores @ 2.0 GHz) |
| **GPU** | NVIDIA A100 PCIe (40 GB HBM2) |
| **GPU Compute Capability** | 8.0 (sm_80) |
| **GPU Memory Bandwidth** | 1.6 TB/s |
| **Driver Version** | 545.23.08 |
| **CUDA Toolkit Version** | 12.9 |
| **Operating System** | Linux (x86_64, AlmaLinux 8) |

---

## 7. Performance Notes

- The simulation achieves **tens of millions of GBM paths in milliseconds**, showcasing the scalability of embarrassingly parallel Monte Carlo workloads on modern GPUs.
- **cuRAND** enables statistically robust Gaussian random number generation.
- Memory access is coalesced to maximize throughput.
- Kernel occupancy and block size (typically 256 threads) were optimized for the A100 architecture.

---

## 8. Validation

Theoretical expectation:

$$E[S_T] = S_0 e^{\mu T} = 105.127$$

$$SD[S_T] = S_0 e^{\mu T}\sqrt{e^{\sigma^2T} - 1} = 21.27$$

Simulation results:
| Quantity | Theoretical | Simulated | Error |
|-----------|--------------|------------|--------|
| **Mean** | 105.127 | 105.139 | +0.01% |
| **StdDev** | 21.27 | 21.25 | -0.09% |

Results confirm near-perfect numerical fidelity.

---

## 9. Future Extensions

- **Option Pricing** (European, Asian, Barrier options)
- **Variance Reduction** (Antithetic, Control Variates)
- **Double Precision Benchmarking**
- **Multi-GPU Scaling** with CUDA-aware MPI
- **Integration with PyTorch / CuPy** for ML-based stochastic modeling

---

## 10. Citation

If you use or modify this project in academic or professional work, please cite:

```bibtex  
@misc{monte-carlo-gbm-stock-prices-cuda,
  author = {Shadman, Nabil},
  title = {GPU-Accelerated Monte Carlo Simulation of Stock Prices using Geometric Brownian Motion},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/nabilshadman/monte-carlo-gbm-stock-prices-cuda}
}
```
