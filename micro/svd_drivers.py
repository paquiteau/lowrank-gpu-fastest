"""Micro-benchmark: every way we tried to get the spectrum of B tall-skinny matrices.

    uv run python micro/svd_drivers.py

Times only the factorization of a centered (B, N, T) float32 batch -- what
each driver costs on its own, before any denoising logic is wrapped around it.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from step5_xsyev_fp32 import XsyevBatched  # noqa: E402

B, N, T = int(os.environ.get("BATCH", "512")), 11**3, 48


def bench(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1e3 / B  # us / patch


@torch.inference_mode()
def main():
    torch.manual_seed(0)
    x = torch.randn(B, N, T, device="cuda") + 1000.0
    xc = x - x.mean(2, keepdim=True)  # temporal centering
    g = xc.mT @ xc
    solver = XsyevBatched(B, T)

    cases = {
        "torch.linalg.svd, default driver": lambda: torch.linalg.svd(
            xc, full_matrices=False
        ),
        "torch.linalg.svd, driver='gesvd'": lambda: torch.linalg.svd(
            xc, full_matrices=False, driver="gesvd"
        ),
        "torch.linalg.svd, driver='gesvdj'": lambda: torch.linalg.svd(
            xc, full_matrices=False, driver="gesvdj"
        ),
        "torch.linalg.svd, driver='gesvda'": lambda: torch.linalg.svd(
            xc, full_matrices=False, driver="gesvda"
        ),
        "Gram + torch.linalg.eigh": lambda: torch.linalg.eigh(xc.mT @ xc),
        "Gram + cusolverDnXsyevBatched (FP32)": lambda: solver(xc.mT @ xc),
        "  of which: Gram GEMM only": lambda: xc.mT @ xc,
        "  of which: XsyevBatched only": lambda: solver(g.clone()),
    }
    print(f"GPU: {torch.cuda.get_device_name()}   batch ({B}, {N}, {T}) float32\n")
    print(f"| {'method':<38} | {'us/patch':>9} |")
    print(f"|{'-' * 40}|{'-' * 11}|")
    for name, fn in cases.items():
        print(f"| {name:<38} | {bench(fn):9.2f} |", flush=True)


if __name__ == "__main__":
    main()
