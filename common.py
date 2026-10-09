"""Shared harness: synthetic data, problem configuration, timing and validation.

Nothing in here is optimized: every rung of the ladder (``stepN_*.py``) is a
complete, self-contained reimplementation of the same denoiser and only borrows
the problem definition and the measuring tape from this module.

The denoiser is MP-PCA (Veraart et al. 2016) with weighted recombination:

1. slide a ``(px, py, pz)`` window over the volume with stride ``STEP``;
2. each patch is a Casorati matrix ``X`` of shape ``(N, T)``, N = px*py*pz;
3. center it along time (subtract each voxel's temporal mean), take the
   SVD, drop the null direction this creates (the constant time course) and
   keep the ``p`` components above the Marchenko-Pastur edge of the
   remaining N x (T - 1) noise;
4. add ``w * X_denoised`` back into the volume with ``w = 1 / (2 + p)``,
   and divide by the accumulated weights at the end.
"""

import itertools
import json
import os
import time
from pathlib import Path

import numpy as np

VOL_SHAPE = tuple(int(s) for s in os.environ.get("VOL", "96,96,48,48").split(","))
PATCH = (11, 11, 11)
STEP = int(os.environ.get("STEP", "3"))
T = VOL_SHAPE[-1]
N = int(np.prod(PATCH))

RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)


def patch_corners(vol_shape=VOL_SHAPE, patch=PATCH, step=STEP) -> np.ndarray:
    """Top-left corners of every patch, shape ``(n_patches, 3)``."""
    # add a last, shifted patch where the stride does not land on the border
    ranges = [
        sorted({*range(0, s - p + 1, step), s - p})
        for s, p in zip(vol_shape[:3], patch)
    ]
    return np.array(list(itertools.product(*ranges)), dtype=np.int64)


def make_phantom(shape=VOL_SHAPE, rank=5, sigma=4.0, seed=0):
    """Low-rank spatio-temporal signal on a strong baseline, plus Gaussian noise.

    Returns ``(noisy, clean)``, both float32 of shape ``shape``. The baseline
    (~1000) is deliberate: real magnitude MRI is never zero-mean, and anything
    that skips proper centering gets punished for it.
    """
    rng = np.random.default_rng(seed)
    *spatial, t = shape
    grids = np.meshgrid(*(np.linspace(0, 1, s) for s in spatial), indexing="ij")

    def smooth_field():
        freqs = rng.uniform(0.5, 3.0, size=3)
        phases = rng.uniform(0, 2 * np.pi, size=3)
        return np.prod(
            [np.cos(2 * np.pi * f * g + ph) for f, g, ph in zip(freqs, grids, phases)],
            axis=0,
        )

    spatial_maps = np.stack([smooth_field() for _ in range(rank)], axis=-1)
    time_axis = np.linspace(0, 1, t)
    courses = np.stack(
        [
            np.sin(2 * np.pi * rng.uniform(1, 6) * time_axis + rng.uniform(0, 6))
            for _ in range(rank)
        ]
    )
    baseline = 1000.0 * (1.2 + smooth_field())[..., None]
    clean = baseline + 20.0 * spatial_maps @ courses
    noisy = clean + rng.normal(scale=sigma, size=shape)
    return noisy.astype(np.float32), clean.astype(np.float32)


def rmse(x, ref):
    return float(np.sqrt(np.mean((x.astype(np.float64) - ref) ** 2)))


def _sync():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def _torch_version():
    try:
        import torch

        return torch.__version__
    except ImportError:
        return None


def run(name, denoise, repeats=5, max_seconds=60.0):
    """Benchmark ``denoise(noisy) -> denoised`` end-to-end and validate it.

    Timing is wall clock around the whole call (host->device copy included),
    after one warm-up call that also absorbs JIT/compile/allocator costs. The
    output is checked against the step 0 reference.
    """
    name += os.environ.get("SUFFIX", "")
    noisy, clean = make_phantom()
    n_patches = len(patch_corners())
    print(
        f"[{name}] volume {VOL_SHAPE}, patch {PATCH}x{T}, step {STEP}, "
        f"{n_patches} patches"
    )

    if os.environ.get("NSYS"):  # profile exactly one warm call, see profile.sh
        import torch

        denoise(noisy)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        denoise(noisy)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        return

    t0 = time.perf_counter()
    out = denoise(noisy)
    _sync()
    first = time.perf_counter() - t0

    times = []
    budget = time.perf_counter() + max_seconds
    for _ in range(repeats):
        t0 = time.perf_counter()
        denoise(noisy)
        _sync()
        times.append(time.perf_counter() - t0)
        if time.perf_counter() > budget:
            break
    best = min(times)

    ref_path = RESULTS / f"ref_{'x'.join(map(str, VOL_SHAPE))}_s{STEP}.npy"
    if name.startswith("step0"):
        np.save(ref_path, out)
    ref = np.load(ref_path) if ref_path.exists() else out
    rel_err = np.linalg.norm(out - ref) / np.linalg.norm(ref)

    res = dict(
        name=name,
        torch=_torch_version(),
        seconds=best,
        first_call=first,
        us_per_patch=1e6 * best / n_patches,
        rel_err_vs_step0=float(rel_err),
        rmse_noisy=rmse(noisy, clean),
        rmse_denoised=rmse(out, clean),
        n_patches=n_patches,
    )
    print(
        f"  best {best * 1e3:9.1f} ms   ({res['us_per_patch']:8.2f} us/patch)   "
        f"first call {first * 1e3:.0f} ms"
    )
    print(
        f"  RMSE noisy {res['rmse_noisy']:.3f} -> denoised "
        f"{res['rmse_denoised']:.3f},  rel. err vs step0 {rel_err:.1e}"
    )
    tag = "x".join(map(str, VOL_SHAPE)) + f"_s{STEP}"
    (RESULTS / f"{name}_{tag}.json").write_text(json.dumps(res, indent=1))
    return res
