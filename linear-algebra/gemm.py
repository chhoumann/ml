#!/usr/bin/env python3
import time
import numpy as np

N = 4096

if __name__ == "__main__":
    # N^2 in memory
    A = np.random.randn(N, N).astype(np.float32)

    # N^2 in memory
    B = np.random.randn(N, N).astype(np.float32)

    # N^2 output cells with 2N compute each. This is the amount of computations we do for
    # matrix multiplication.
    flop = N * N * 2 * N

    st = time.monotonic()
    C = A @ B
    et = time.monotonic()
    s = et - st

    print(f"{flop/s * 1e-12:.2f} TFLOP/S")

    # My machine has a Intel(R) Core(TM) i7-8700
    # You can compute theoretical peak FLOPS by:
    # - CPU frequency (GHz), which is the clock speed of your CPU. It's the number of cycles it can execute per second.
    # - Number of cores: the number of processing units in your CPU.
    # Number of operations per cycle, which depends on your CPU architecture. E.g. CPUs with AVX (Advanced Vector Extensions) can do 8 ops per cycle, while a CPPU with AVX-512 can do 16 ops per cycle.
    # Then:
    # FLOPS = CPU frequency * number of cores * number of operations per cycle
    cpu_freq = 4.7  # GHz - overclocked
    num_cores = 6  # with 12 threads
    ops_per_cycle = 8  # avx2
    # The below will be GFLOPS because CPU frequency is measured in GHz.
    print(f"Max GFLOPS: {cpu_freq*num_cores*ops_per_cycle}")
    # Although the result seems odd, as I seem to reach practical measured FLOPS above the theoretical limit??
