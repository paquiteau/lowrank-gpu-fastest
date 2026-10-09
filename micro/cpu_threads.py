"""Micro-benchmark: one (1331, 48) SVD in NumPy vs the number of BLAS threads.

for t in 1 4 16 64; do OPENBLAS_NUM_THREADS=$t uv run python micro/cpu_threads.py; done
"""

import os
import timeit

import numpy as np

x = np.random.default_rng(0).standard_normal((11**3, 48)).astype(np.float32)
n, reps = 200, 5
best = min(
    timeit.repeat(lambda: np.linalg.svd(x, full_matrices=False), number=n, repeat=reps)
)
print(
    f"OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS', 'unset'):>5}: "
    f"{best / n * 1e6:8.1f} us per SVD"
)
