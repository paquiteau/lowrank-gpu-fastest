"""Step 4 -- do gesvda's job ourselves: Gram matrix, eigh, and never form U.

For a centered (N, T) patch with N >> T, the right singular vectors and the
singular values come from the small (T, T) Gram matrix:

    Xc^T Xc = V diag(s^2) V^T.

Patches are centered along time, so Xc 1 = 0: the constant time course is an
eigenvector with eigenvalue 0 (the smallest), which ``mp_keep`` drops.

MP-PCA keeps the top-p components, so the denoised patch is

    U_p S_p V_p^T + m = Xc V_p V_p^T + m = Xc P + m,

with P = V_p V_p^T a (T, T) projector. U is never needed: one small eigh and
one GEMM per patch, fused with the ``+ m`` through ``baddbmm``.
"""

import os

import torch

from common import PATCH, patch_corners, run

BATCH = int(os.environ.get("BATCH", "512"))


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
    lam, v = torch.linalg.eigh(xc.mT @ xc)  # ascending, V in columns
    keep, p = mp_keep(lam, n)
    proj = (v * keep[:, None, :]) @ v.mT  # V_p V_p^T
    return torch.baddbmm(m, xc, proj), p


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
            x = flat[idx]
        with torch.cuda.nvtx.range("mppca"):
            xd, p = mppca_batch(x)
        with torch.cuda.nvtx.range("scatter"):
            w = 1.0 / (2.0 + p.float())
            out.index_add_(0, idx.reshape(-1), (xd * w[:, None, None]).reshape(-1, t))
            wsum.index_add_(0, idx.reshape(-1), w.repeat_interleave(idx.shape[1]))
    return (out / wsum.clamp_min(1e-12)[:, None]).view(vol.shape).cpu().numpy()


if __name__ == "__main__":
    run("step4_gram_eigh", denoise)
