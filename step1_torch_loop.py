"""Step 1 -- the naive GPU port: same loop, ``np`` -> ``torch``, ``.cuda()``.

The volume is uploaded once, every patch is still processed on its own.
"""

import torch

from common import PATCH, patch_corners, run


def mppca_patch(x):
    n, t = x.shape
    m = x.mean(dim=1, keepdim=True)  # temporal mean of each voxel
    u, s, vt = torch.linalg.svd(x - m, full_matrices=False)
    t -= 1  # drop the null direction (constant time course)
    u, s, vt = u[:, :t], s[:t], vt[:t]

    eigs = s**2 / n
    rcum = eigs.flip(0).cumsum(0).flip(0)
    p_range = torch.arange(t, device=x.device)
    signal = (eigs - eigs[-1]) * (t - p_range) * (n - p_range) > 4 * rcum * (
        t * n
    ) ** 0.5
    p = int(signal.sum())  # <- device -> host sync, every patch
    return (u[:, :p] * s[:p]) @ vt[:p] + m, p


@torch.inference_mode()
def denoise(vol_np):
    vol = torch.from_numpy(vol_np).cuda()
    out = torch.zeros_like(vol)
    wsum = torch.zeros(vol.shape[:3] + (1,), device="cuda")
    px, py, pz = PATCH
    for i, j, k in patch_corners():
        sl = (slice(i, i + px), slice(j, j + py), slice(k, k + pz))
        x = vol[sl].reshape(-1, vol.shape[-1])
        with torch.cuda.nvtx.range("patch"):
            xd, p = mppca_patch(x)
        w = 1.0 / (2.0 + p)
        out[sl] += w * xd.reshape(px, py, pz, -1)
        wsum[sl] += w
    return (out / wsum.clamp_min(1e-12)).cpu().numpy()


if __name__ == "__main__":
    run("step1_torch_loop", denoise, repeats=2)
