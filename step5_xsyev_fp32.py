"""Step 5 -- call cuSOLVER ourselves: a batched eigensolver that really runs in FP32.

Profiling step 3 shows ``gesvda`` launching ``double`` kernels on float32 input:
cuSOLVER promotes its Gram/eig/back-transform to FP64 internally, and consumer
or workstation GPUs run FP64 at 1/64 of the FP32 rate. ``torch.linalg.eigh``
(step 4) stays in FP32 but uses a Jacobi solver tuned for tiny matrices.

``cusolverDnXsyevBatched`` (CUDA 12.6+) is a batched divide-and-conquer
eigensolver with an explicit compute type. PyTorch does not expose it; the
``nvmath-python`` bindings, generated from the cuSOLVER headers, do.
"""

import os

import nvmath.bindings.cublas as cublas
import nvmath.bindings.cusolver as cusolver
import nvmath.bindings.cusolverDn as cusolverDn
import torch

from common import PATCH, patch_corners, run

BATCH = int(os.environ.get("BATCH", "512"))
CUDA_R_32F = 0


class XsyevBatched:
    """Batched FP32 symmetric eigensolver for a fixed (batch, T), workspace cached.

    ``solver(g)`` overwrites ``g`` (B, T, T) with eigenvectors and returns the
    ascending eigenvalues (B, T). cuSOLVER is column-major, so read row-major
    the eigenvectors come back as rows: ``g[b, j]`` is eigenvector j.
    """

    def __init__(self, batch, t, device="cuda"):
        self.batch, self.t = batch, t
        self.handle = cusolverDn.create()
        self.params = cusolverDn.create_params()
        g = torch.empty(batch, t, t, device=device)
        self.w = torch.empty(batch, t, device=device)
        self.info = torch.empty(batch, dtype=torch.int32, device=device)
        args = (cusolver.EigMode.VECTOR, cublas.FillMode.LOWER, t, CUDA_R_32F)
        dev_bytes, host_bytes = cusolverDn.xsyev_batched_buffer_size(
            self.handle, self.params, *args, g.data_ptr(), t,
            CUDA_R_32F, self.w.data_ptr(), CUDA_R_32F, batch,
        )  # fmt: skip
        self.dev_ws = torch.empty(max(dev_bytes, 1), dtype=torch.uint8, device=device)
        self.host_ws = torch.empty(max(host_bytes, 1), dtype=torch.uint8)
        self._args = args

    def __call__(self, g):
        assert g.shape == (self.batch, self.t, self.t) and g.is_contiguous()
        cusolverDn.set_stream(self.handle, torch.cuda.current_stream().cuda_stream)
        cusolverDn.xsyev_batched(
            self.handle, self.params, *self._args, g.data_ptr(), self.t,
            CUDA_R_32F, self.w.data_ptr(), CUDA_R_32F,
            self.dev_ws.data_ptr(), self.dev_ws.numel(),
            self.host_ws.data_ptr(), self.host_ws.numel(),
            self.info.data_ptr(), self.batch,
        )  # fmt: skip
        return self.w

    def __del__(self):
        cusolverDn.destroy_params(self.params)
        cusolverDn.destroy(self.handle)


_solvers: dict[tuple[int, int], XsyevBatched] = {}


def eigh_fp32(g):
    key = (g.shape[0], g.shape[-1])
    if key not in _solvers:  # one workspace per batch size (the last batch is short)
        _solvers[key] = XsyevBatched(*key)
    return _solvers[key](g)


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
    g = xc.mT @ xc
    lam = eigh_fp32(g)  # g now holds V^T (eigenvectors as rows)
    keep, p = mp_keep(lam, n)
    proj = g.mT @ (keep[:, :, None] * g)  # V_p V_p^T
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
    run("step5_xsyev_fp32", denoise)
