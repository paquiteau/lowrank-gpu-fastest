"""Step 8 -- polish: tune the launch configuration, pin the output buffer.

Same two kernels as step 7. Two measured fixes:

* ``recon_scatter_kernel`` was launched with Triton's defaults. A sweep over
  (BLOCK_N, num_warps) finds 2.5x on that kernel: smaller tiles mean more
  programs in flight to hide the latency of the atomics.
* The final ``.cpu()`` copies 85 MB into *pageable* memory, which the driver
  stages through a bounce buffer at ~2.5 GB/s. Copying into a pinned buffer
  runs at full PCIe speed -- 37 ms -> 3 ms, a third of the whole step 7 runtime.
"""

import os

import torch
import triton
import triton.language as tl

from common import PATCH, patch_corners, run

BATCH = int(os.environ.get("BATCH", "2048"))  # only (B, T, T) tensors live per batch
GRAM_CFG = dict(BLOCK_N=32, num_warps=4)
RECON_CFG = dict(BLOCK_N=32, num_warps=8)  # from micro/tune_recon.py


@triton.jit
def _patch_rows(base, n, ny, nz, PY: tl.constexpr, PZ: tl.constexpr):
    """Voxel ``n`` of a patch whose first voxel is ``base`` -> volume row."""
    return base + (n // (PY * PZ)) * ny * nz + (n // PZ % PY) * nz + n % PZ


@triton.jit
def gram_kernel(
    vol_ptr, base_ptr, gram_ptr,
    ny, nz, T,
    PX: tl.constexpr, PY: tl.constexpr, PZ: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_T: tl.constexpr,
):  # fmt: skip
    b = tl.program_id(0)
    base = tl.load(base_ptr + b)
    t = tl.arange(0, BLOCK_T)
    t_mask = t < T
    N: tl.constexpr = PX * PY * PZ

    gram = tl.zeros((BLOCK_T, BLOCK_T), dtype=tl.float32)
    for n0 in range(0, N, BLOCK_N):
        n = n0 + tl.arange(0, BLOCK_N)
        mask = (n < N)[:, None] & t_mask[None, :]
        row = _patch_rows(base, n, ny, nz, PY, PZ)
        x = tl.load(vol_ptr + row[:, None] * T + t[None, :], mask=mask, other=0.0)
        m = tl.sum(x, 1) / T  # temporal mean of each voxel: a row reduction
        xc = tl.where(mask, x - m[:, None], 0.0)
        gram = tl.dot(tl.trans(xc), xc, gram, input_precision="ieee")

    g_ptrs = gram_ptr + b * T * T + t[:, None] * T + t[None, :]
    tl.store(g_ptrs, gram, mask=t_mask[:, None] & t_mask[None, :])


@triton.jit
def recon_scatter_kernel(
    vol_ptr, base_ptr, proj_ptr, w_ptr, out_ptr, wsum_ptr,
    ny, nz, T,
    PX: tl.constexpr, PY: tl.constexpr, PZ: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_T: tl.constexpr,
):  # fmt: skip
    b = tl.program_id(0)
    n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    t = tl.arange(0, BLOCK_T)
    n_mask = n < PX * PY * PZ
    t_mask = t < T
    mask = n_mask[:, None] & t_mask[None, :]

    row = _patch_rows(tl.load(base_ptr + b), n, ny, nz, PY, PZ)
    proj = tl.load(
        proj_ptr + b * T * T + t[:, None] * T + t[None, :],
        mask=t_mask[:, None] & t_mask[None, :],
        other=0.0,
    )
    w = tl.load(w_ptr + b)

    x = tl.load(vol_ptr + row[:, None] * T + t[None, :], mask=mask, other=0.0)
    m = tl.sum(x, 1) / T
    xc = tl.where(mask, x - m[:, None], 0.0)
    xd = tl.dot(xc, proj, input_precision="ieee") + m[:, None]
    tl.atomic_add(out_ptr + row[:, None] * T + t[None, :], w * xd, mask=mask)
    tl.atomic_add(wsum_ptr + row, w + tl.zeros_like(row).to(tl.float32), mask=n_mask)


def patch_bases(vol_shape, device):
    """Row of the first voxel of every patch in the (X*Y*Z, T) view."""
    _, ny, nz, _ = vol_shape
    corners = torch.from_numpy(patch_corners()).to(device)
    return (corners * torch.tensor([ny * nz, nz, 1], device=device)).sum(-1)


def mp_keep(lam, n):
    """MP-PCA rank from ascending Gram eigenvalues ``lam`` (B, T).

    ``lam[:, 0]`` belongs to the constant time course, a null direction once
    patches are centered along time: never kept, and not part of the noise
    bulk, which is an N x (T - 1) matrix. Returns the (B, T) 0/1 mask of kept
    components (in ascending order) and p.
    """
    t = lam.shape[-1] - 1
    eigs = lam[:, 1:].flip(-1).clamp_min(0) / n  # descending, null dropped
    rcum = eigs.flip(-1).cumsum(-1).flip(-1)
    p_range = torch.arange(t, device=lam.device)
    signal = (eigs - eigs[:, -1:]) * (t - p_range) * (n - p_range) > 4 * rcum * (
        t * n
    ) ** 0.5
    p = signal.sum(-1)
    keep = (p_range < p[:, None]).flip(-1).to(lam.dtype)
    return torch.nn.functional.pad(keep, (1, 0)), p


@torch.inference_mode()
def denoise(vol_np):
    vol = torch.from_numpy(vol_np).cuda()
    _, ny, nz, t = vol.shape
    n = PATCH[0] * PATCH[1] * PATCH[2]
    out = torch.zeros_like(vol)
    wsum = torch.zeros(vol.shape[:3], device="cuda")
    block_t = max(triton.next_power_of_2(t), 16)
    meta = dict(PX=PATCH[0], PY=PATCH[1], PZ=PATCH[2], BLOCK_T=block_t)

    bases = patch_bases(vol.shape, "cuda")
    for b0 in range(0, bases.shape[0], BATCH):
        base = bases[b0 : b0 + BATCH]
        nb = base.shape[0]
        gram = torch.empty(nb, t, t, device="cuda")
        with torch.cuda.nvtx.range("gram"):
            gram_kernel[(nb,)](vol, base, gram, ny, nz, t, **meta, **GRAM_CFG)
        with torch.cuda.nvtx.range("eigh"):
            lam, v = torch.linalg.eigh(gram)
            keep, p = mp_keep(lam, n)
            proj = (v * keep[:, None, :]) @ v.mT
            w = 1.0 / (2.0 + p.float())
        with torch.cuda.nvtx.range("recon+scatter"):
            grid = (nb, triton.cdiv(n, RECON_CFG["BLOCK_N"]))
            recon_scatter_kernel[grid](
                vol, base, proj.contiguous(), w, out, wsum, ny, nz, t,
                **meta, **RECON_CFG,
            )  # fmt: skip
    host = torch.empty(out.shape, pin_memory=True)  # cached by torch's host allocator
    host.copy_(out / wsum.clamp_min(1e-12)[..., None])
    return host.numpy()


if __name__ == "__main__":
    run("step8_tuned", denoise)
