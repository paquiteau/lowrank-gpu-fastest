"""Step 6 -- a Triton kernel for the weighted scatter-add.

Step 5 spends a surprising share of its time *recombining*: build a (B*N,)
int64 index tensor, multiply the batch by its weights (one full read+write of
the batch), ``index_add_`` the values, ``repeat_interleave`` the weights and
``index_add_`` them too. The kernel below does all of it in a single pass:
patch voxel addresses are recomputed from the patch corner in registers, the
weight is applied on the fly, and both accumulators are updated with atomics.

The eigensolver is back to ``torch.linalg.eigh``: PyTorch 2.14 (not yet 2.11)
dispatches batched FP32 input to the same ``syevBatched`` kernels step 5 binds
by hand (on older versions, import ``eigh_fp32`` from step 5 instead).
"""

import os

import torch
import triton
import triton.language as tl

from common import PATCH, patch_corners, run

BATCH = int(os.environ.get("BATCH", "512"))


@triton.jit
def scatter_add_kernel(
    out_ptr, wsum_ptr, xd_ptr, w_ptr, base_ptr,
    ny, nz, T,
    PX: tl.constexpr, PY: tl.constexpr, PZ: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_T: tl.constexpr,
):  # fmt: skip
    """out[rows] += w * xd[b], wsum[rows] += w, for patch b and a block of rows."""
    b = tl.program_id(0)
    n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    t = tl.arange(0, BLOCK_T)
    n_mask = n < PX * PY * PZ
    mask = n_mask[:, None] & (t < T)[None, :]

    # voxel n of the patch -> row of the (X*Y*Z, T) volume
    row = tl.load(base_ptr + b) + (n // (PY * PZ)) * ny * nz + (n // PZ % PY) * nz
    row += n % PZ
    w = tl.load(w_ptr + b)

    xd = tl.load(xd_ptr + (b * PX * PY * PZ + n[:, None]) * T + t[None, :], mask=mask)
    tl.atomic_add(out_ptr + row[:, None] * T + t[None, :], w * xd, mask=mask)
    tl.atomic_add(wsum_ptr + row, w + tl.zeros_like(row).to(tl.float32), mask=n_mask)


def scatter_add(out, wsum, xd, w, base, vol_shape):
    b, n, t = xd.shape
    _, ny, nz, _ = vol_shape
    block_n = 64
    scatter_add_kernel[(b, triton.cdiv(n, block_n))](
        out, wsum, xd, w, base, ny, nz, t, *PATCH,
        BLOCK_N=block_n, BLOCK_T=triton.next_power_of_2(t),
    )  # fmt: skip


def patch_bases(vol_shape, device):
    """Row of the first voxel of every patch in the (X*Y*Z, T) view."""
    _, ny, nz, _ = vol_shape
    corners = torch.from_numpy(patch_corners()).to(device)
    return (corners * torch.tensor([ny * nz, nz, 1], device=device)).sum(-1)


def patch_offsets(vol_shape, device):
    _, ny, nz, _ = vol_shape
    ox, oy, oz = torch.meshgrid(
        *(torch.arange(p, device=device) for p in PATCH), indexing="ij"
    )
    return (ox * ny * nz + oy * nz + oz).reshape(-1)


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


def mppca_batch(x):
    _, n, _ = x.shape
    m = x.mean(dim=2, keepdim=True)  # temporal mean of each voxel
    xc = x - m
    lam, v = torch.linalg.eigh(xc.mT @ xc)
    keep, p = mp_keep(lam, n)
    proj = (v * keep[:, None, :]) @ v.mT
    return torch.baddbmm(m, xc, proj), p


@torch.inference_mode()
def denoise(vol_np):
    vol = torch.from_numpy(vol_np).cuda()
    t = vol.shape[-1]
    flat = vol.view(-1, t)
    out = torch.zeros_like(flat)
    wsum = torch.zeros(flat.shape[0], device="cuda")

    bases = patch_bases(vol.shape, "cuda")
    offsets = patch_offsets(vol.shape, "cuda")
    for b in range(0, bases.shape[0], BATCH):
        base = bases[b : b + BATCH]
        with torch.cuda.nvtx.range("gather"):
            x = flat[base[:, None] + offsets]
        with torch.cuda.nvtx.range("mppca"):
            xd, p = mppca_batch(x)
        with torch.cuda.nvtx.range("scatter"):
            scatter_add(out, wsum, xd, 1.0 / (2.0 + p.float()), base, vol.shape)
    return (out / wsum.clamp_min(1e-12)[:, None]).view(vol.shape).cpu().numpy()


if __name__ == "__main__":
    run("step6_triton_scatter", denoise)
