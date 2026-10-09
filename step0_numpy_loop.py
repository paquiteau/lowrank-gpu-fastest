"""Step 0 -- the textbook implementation: a Python loop over patches, NumPy SVD."""

import numpy as np

from common import PATCH, patch_corners, run


def mppca_patch(x):
    """MP-PCA on one (N, T) Casorati matrix. Returns (denoised, rank)."""
    n, t = x.shape
    m = x.mean(axis=1, keepdims=True)  # temporal mean of each voxel
    u, s, vt = np.linalg.svd(x - m, full_matrices=False)
    # Centering along time makes the constant time course a null direction:
    # drop it, the noise is an N x (T - 1) matrix.
    t -= 1
    u, s, vt = u[:, :t], s[:t], vt[:t]

    eigs = s**2 / n  # descending
    rcum = np.cumsum(eigs[::-1])[::-1]  # rcum[i] = sum_{j >= i} eigs[j]
    p_range = np.arange(t)
    signal = (eigs - eigs[-1]) * (t - p_range) * (n - p_range) > 4 * rcum * np.sqrt(
        t * n
    )
    p = int(signal.sum())
    return (u[:, :p] * s[:p]) @ vt[:p] + m, p


def denoise(vol):
    out = np.zeros_like(vol)
    wsum = np.zeros(vol.shape[:3] + (1,), dtype=vol.dtype)
    px, py, pz = PATCH
    for i, j, k in patch_corners():
        sl = np.s_[i : i + px, j : j + py, k : k + pz]
        x = vol[sl].reshape(-1, vol.shape[-1])
        xd, p = mppca_patch(x)
        w = 1.0 / (2.0 + p)
        out[sl] += w * xd.reshape(px, py, pz, -1)
        wsum[sl] += w
    return out / np.maximum(wsum, 1e-12)


if __name__ == "__main__":
    run("step0_numpy_loop", denoise, repeats=2)
