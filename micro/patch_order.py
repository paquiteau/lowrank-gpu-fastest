"""Micro-benchmark: patch order within batches, and device-to-host copies.

Step 7 with patches in grid order, "coloured" so that no two patches of a
batch overlap (no atomic contention), and shuffled. Then the final copy back to
the host, into pageable vs pinned memory.

    uv run python micro/patch_order.py
"""

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import common  # noqa: E402
import step7_triton_fused as s7  # noqa: E402


def best_ms(fn, n=5):
    fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return 1e3 * min(times)


def main():
    noisy, _ = common.make_phantom()
    corners = common.patch_corners()
    k = -(-common.PATCH[0] // common.STEP)  # patches closer than k strides overlap
    colour = (corners // common.STEP) % k
    orders = {
        "grid order": np.arange(len(corners)),
        "coloured (no overlap in a batch)": np.lexsort(
            (corners[:, 2], corners[:, 1], corners[:, 0], *colour.T[::-1])
        ),
        "shuffled": np.random.default_rng(0).permutation(len(corners)),
    }
    print("| step 7, patch order | ms |")
    print("|---|---|")
    for name, order in orders.items():
        s7.patch_corners = lambda order=order: corners[order]
        print(f"| {name} | {best_ms(lambda: s7.denoise(noisy)):.1f} |")

    x = torch.randn(noisy.shape, device="cuda")
    pinned = torch.empty(noisy.shape, pin_memory=True)
    print("\n| copy | ms |")
    print("|---|---|")
    print(f"| out.cpu() (pageable) | {best_ms(lambda: x.cpu()):.1f} |")
    print(f"| pinned.copy_(out) | {best_ms(lambda: pinned.copy_(x)):.1f} |")
    fresh = best_ms(lambda: torch.empty(noisy.shape, pin_memory=True).copy_(x))
    print(f"| torch.empty(..., pin_memory=True).copy_(out) | {fresh:.1f} |")


if __name__ == "__main__":
    main()
