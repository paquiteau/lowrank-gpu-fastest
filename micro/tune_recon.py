"""Micro-benchmark: launch configuration of step 7's ``recon_scatter_kernel``.

Also times a variant that writes with plain stores instead of atomics (wrong
result, same arithmetic and loads) to see what the atomics themselves cost.

    uv run python micro/tune_recon.py
"""

import os
import sys

import torch
import triton
import triton.language as tl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import step7_triton_fused as s7  # noqa: E402
from common import PATCH, make_phantom  # noqa: E402


@triton.jit
def recon_store_kernel(
    vol_ptr, base_ptr, proj_ptr, w_ptr, out_ptr,
    ny, nz, T,
    PX: tl.constexpr, PY: tl.constexpr, PZ: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_T: tl.constexpr,
):  # fmt: skip
    b = tl.program_id(0)
    n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    t = tl.arange(0, BLOCK_T)
    t_mask = t < T
    mask = (n < PX * PY * PZ)[:, None] & t_mask[None, :]
    row = s7._patch_rows(tl.load(base_ptr + b), n, ny, nz, PY, PZ)
    proj = tl.load(
        proj_ptr + b * T * T + t[:, None] * T + t[None, :],
        mask=t_mask[:, None] & t_mask[None, :],
        other=0.0,
    )
    x = tl.load(vol_ptr + row[:, None] * T + t[None, :], mask=mask, other=0.0)
    m = tl.sum(x, 1) / T
    xc = tl.where(mask, x - m[:, None], 0.0)
    xd = tl.dot(xc, proj, input_precision="ieee") + m[:, None]
    out = out_ptr + (b * PX * PY * PZ + n[:, None]) * T + t[None, :]
    tl.store(out, tl.load(w_ptr + b) * xd, mask=mask)


def main(nb=512):
    noisy, _ = make_phantom()
    vol = torch.from_numpy(noisy).cuda()
    _, ny, nz, t = vol.shape
    n = PATCH[0] * PATCH[1] * PATCH[2]
    base = s7.patch_bases(vol.shape, "cuda")[:nb]
    proj = torch.eye(t, device="cuda").expand(nb, t, t).contiguous()
    w = torch.rand(nb, device="cuda")
    out, wsum = torch.zeros_like(vol), torch.zeros(vol.shape[:3], device="cuda")
    scratch = torch.empty(nb, n, t, device="cuda")

    print(f"one batch of {nb} patches; times in us\n")
    print("| BLOCK_N | num_warps | atomic_add | plain store |")
    print("|---------|-----------|------------|-------------|")
    for block_n in (32, 64, 128):
        for warps in (2, 4, 8):
            meta = dict(PX=PATCH[0], PY=PATCH[1], PZ=PATCH[2], BLOCK_N=block_n,
                        BLOCK_T=triton.next_power_of_2(t), num_warps=warps)  # fmt: skip
            grid = (nb, triton.cdiv(n, block_n))
            atomic = triton.testing.do_bench(lambda: s7.recon_scatter_kernel[grid](
                vol, base, proj, w, out, wsum, ny, nz, t, **meta))  # fmt: skip
            store = triton.testing.do_bench(lambda: recon_store_kernel[grid](
                vol, base, proj, w, scratch, ny, nz, t, **meta))  # fmt: skip
            print(f"| {block_n:7d} | {warps:9d} | {atomic * 1e3:10.0f} | {store * 1e3:11.0f} |",
                  flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
