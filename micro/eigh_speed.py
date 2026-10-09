"""Micro-benchmark: FP32 batched eigh is fast on a badly conditioned G.

Divide and conquer skips ("deflates") every eigenvalue negligible relative to
||G||. With spatial centering the mean image makes lambda_1 dominate and the
whole noise spectrum gets skipped -- fast and inaccurate. Temporal centering
removes it, and the solver does the full work.

    uv run python micro/eigh_speed.py
"""

import os
import sys

import torch
import triton

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import PATCH, make_phantom, patch_corners  # noqa: E402


@torch.inference_mode()
def main(nb=2048):
    noisy, _ = make_phantom()
    vol = torch.from_numpy(noisy).cuda()
    px, py, pz = PATCH
    x = torch.stack(
        [vol[i : i + px, j : j + py, k : k + pz].reshape(-1, vol.shape[-1])
         for i, j, k in patch_corners()[:nb]]
    )  # fmt: skip
    xs = x - x.mean(1, keepdim=True)
    xt = x - x.mean(2, keepdim=True)
    noise = torch.randn_like(x)
    cases = {
        "G, spatial centering (FP32)": xs.mT @ xs,
        "G, spatial centering (FP64)": (xs.mT @ xs).double(),
        "G, temporal centering (FP32)": xt.mT @ xt,
        "random SPD (FP32)": noise.mT @ noise,
    }
    print(f"| torch.linalg.eigh, ({nb}, 48, 48) | us/patch |")
    print("|---|---|")
    for name, m in cases.items():
        ms = triton.testing.do_bench(lambda: torch.linalg.eigh(m))
        print(f"| {name} | {ms * 1e3 / nb:.2f} |")


if __name__ == "__main__":
    main()
