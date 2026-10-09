"""Does the Gram-matrix route change any MP-PCA decision? It depends on centering.

Squaring the matrix squares its condition number, the classical argument
against forming X^T X. For MP-PCA what matters is the selected rank p, so
count how often each FP32 route disagrees with an FP64 SVD reference, for the
two ways of centering a patch:

* spatial: subtract the mean over voxels (one value per time point);
* temporal: subtract each voxel's mean over time (the mean image) -- what the
  ladder does.

Then, for spatial centering, isolate where the error enters (Gram matrix vs
eigensolver).

    uv run python micro/rank_agreement.py
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import PATCH, make_phantom, patch_corners  # noqa: E402


def mp_rank(lam, n):
    """MP-PCA rank from descending eigenvalues of Xc^T Xc, (B, M)."""
    t = lam.shape[-1]
    eigs = lam.clamp_min(0) / n
    rcum = eigs.flip(-1).cumsum(-1).flip(-1)
    p_range = torch.arange(t, device=lam.device)
    signal = (eigs - eigs[:, -1:]) * (t - p_range) * (n - p_range) > 4 * rcum * (
        t * n
    ) ** 0.5
    return signal.sum(-1)


def desc(lam_ascending, k):
    """Top-k eigenvalues, descending (drops the null direction when k = T - 1)."""
    return lam_ascending.flip(-1)[:, :k].double()


@torch.inference_mode()
def main(n_patches=4096):
    noisy, _ = make_phantom()
    vol = torch.from_numpy(noisy).cuda()
    px, py, pz = PATCH
    corners = patch_corners()
    pick = np.random.default_rng(0).choice(len(corners), n_patches, replace=False)
    x = torch.stack(
        [vol[i : i + px, j : j + py, k : k + pz].reshape(-1, vol.shape[-1])
         for i, j, k in corners[pick]]
    )  # fmt: skip
    n, t = x.shape[1:]
    x64 = x.double()
    center_t = torch.eye(t, device="cuda") - 1.0 / t  # removes the temporal mean

    print(f"{n_patches} patches of {n} x {t}, MP-PCA rank vs an FP64 SVD reference\n")
    print("| centering | lambda_1 / lambda_min | FP32 route | rank mismatches |")
    print("|---|---|---|---|")
    for name, dim, k in (("spatial", 1, t), ("temporal", 2, t - 1)):
        xc64 = x64 - x64.mean(dim, keepdim=True)
        lam_ref = torch.linalg.svdvals(xc64)[:, :k] ** 2
        p_ref = mp_rank(lam_ref, n)
        cond = (lam_ref[:, 0] / lam_ref[:, -1]).median().item()

        xc = x - x.mean(dim, keepdim=True)
        if dim == 1:
            m = x.mean(1, keepdim=True)
            shortcut = ("X^T X - N m^T m", x.mT @ x - n * m.mT @ m)
        else:
            shortcut = ("C X^T X C", center_t @ (x.mT @ x) @ center_t)
        routes = {
            "SVD of Xc": torch.linalg.svdvals(xc)[:, :k].double() ** 2,
            "eigh(Xc^T Xc)": desc(torch.linalg.eigvalsh(xc.mT @ xc), k),
            f"eigh({shortcut[0]})": desc(torch.linalg.eigvalsh(shortcut[1]), k),
        }
        for route, lam in routes.items():
            bad = (mp_rank(lam, n) != p_ref).sum().item()
            print(f"| {name} | {cond:.1e} | {route} | "
                  f"{bad} ({100 * bad / n_patches:.1f}%) |")  # fmt: skip

    # Spatial centering: which half of the Gram route loses the precision?
    xc64 = x64 - x64.mean(1, keepdim=True)
    p_ref = mp_rank(torch.linalg.svdvals(xc64) ** 2, n)
    xc = x - x.mean(1, keepdim=True)
    g32, g64 = xc.mT @ xc, xc64.mT @ xc64
    xc_np = xc.cpu().numpy()
    g_np = torch.from_numpy(xc_np.transpose(0, 2, 1) @ xc_np).cuda()
    print("\nspatial centering: Gram matrix / eigensolver precision\n")
    print("| Gram matrix | eigensolver | rank mismatches |")
    print("|---|---|---|")
    for gram, solver, g in (
        ("FP32 cuBLAS", "FP32 cuSOLVER", g32),
        ("FP64", "FP32 cuSOLVER", g64.float()),
        ("FP32 cuBLAS", "FP64", g32.double()),
        ("FP32 NumPy (CPU)", "FP64", g_np.double()),
    ):
        bad = (mp_rank(desc(torch.linalg.eigvalsh(g), t), n) != p_ref).sum().item()
        print(f"| {gram} | {solver} | {bad} |")


if __name__ == "__main__":
    main()
