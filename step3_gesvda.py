"""Step 3 -- same as step 2, but ask cuSOLVER for `gesvda`, the batched tall-skinny SVD.

``gesvda`` forms the (T, T) Gram matrix, eigendecomposes it and recovers U:
exactly the right algorithm for N >> T. One keyword argument.
"""

import os

import torch

from common import PATCH, patch_corners, run

BATCH = int(os.environ.get("BATCH", "512"))
SVD_DRIVER = "gesvda"  # approximate SVD for batches of tall-skinny matrices


def patch_row_indices(vol_shape, device):
    """Row index (into the (X*Y*Z, T) view) of every voxel of every patch."""
    _, ny, nz, _ = vol_shape
    corners = torch.from_numpy(patch_corners()).to(device)
    base = (corners * torch.tensor([ny * nz, nz, 1], device=device)).sum(-1)
    ox, oy, oz = torch.meshgrid(
        *(torch.arange(p, device=device) for p in PATCH), indexing="ij"
    )
    offsets = (ox * ny * nz + oy * nz + oz).reshape(-1)
    return base[:, None] + offsets  # (P, N)


def mppca_batch(x):
    """MP-PCA on a (B, N, T) batch. Returns (denoised, rank)."""
    _, n, t = x.shape
    m = x.mean(dim=2, keepdim=True)  # temporal mean of each voxel
    u, s, vh = torch.linalg.svd(x - m, full_matrices=False, driver=SVD_DRIVER)
    t -= 1  # drop the null direction (constant time course)
    u, s, vh = u[..., :t], s[:, :t], vh[:, :t]

    eigs = s**2 / n
    rcum = eigs.flip(-1).cumsum(-1).flip(-1)
    p_range = torch.arange(t, device=x.device)
    signal = (eigs - eigs[:, -1:]) * (t - p_range) * (n - p_range) > 4 * rcum * (
        t * n
    ) ** 0.5
    p = signal.sum(-1)
    s_kept = s * (p_range < p[:, None])
    return torch.baddbmm(m, u * s_kept[:, None, :], vh), p


@torch.inference_mode()
def denoise(vol_np):
    vol = torch.from_numpy(vol_np).cuda()
    t = vol.shape[-1]
    flat = vol.view(-1, t)
    out = torch.zeros_like(flat)
    wsum = torch.zeros(flat.shape[0], device="cuda")

    rows = patch_row_indices(vol.shape, "cuda")
    for b in range(0, rows.shape[0], BATCH):
        idx = rows[b : b + BATCH]
        with torch.cuda.nvtx.range("gather"):
            x = flat[idx]  # (B, N, T)
        with torch.cuda.nvtx.range("mppca"):
            xd, p = mppca_batch(x)
        with torch.cuda.nvtx.range("scatter"):
            w = 1.0 / (2.0 + p.float())
            out.index_add_(0, idx.reshape(-1), (xd * w[:, None, None]).reshape(-1, t))
            wsum.index_add_(0, idx.reshape(-1), w.repeat_interleave(idx.shape[1]))
    return (out / wsum.clamp_min(1e-12)[:, None]).view(vol.shape).cpu().numpy()


if __name__ == "__main__":
    run("step3_gesvda", denoise)
