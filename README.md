# Optimizing SVD of a batch of tall-skinny matrices. 

From a NumPy loop to fused Triton kernels: an optimization ladder for patch-wise
MP-PCA denoising of 4D (3D + time) volumes. 

This repository accompanies the blog post at
<http://pierre-antoine-comby.fr/posts/patch-svd-faster.html>, which walks
through each step and the profiling behind it. The resulting optimization code is used in 
[patch-denoise](https://github.com/paquiteau/patch-denoising)

## The ladder

Each `stepN_*.py` is a complete, self-contained implementation of the same
denoiser; `common.py` only provides the synthetic data, problem definition,
timing and validation against step 0.

| step | file                      | idea                                               |
|------|---------------------------|----------------------------------------------------|
| 0    | `step0_numpy_loop.py`     | Python loop over patches, NumPy SVD                |
| 1    | `step1_torch_loop.py`     | same loop on GPU with PyTorch                      |
| 2    | `step2_torch_batched.py`  | batched gather, SVD and `index_add_`               |
| 3    | `step3_gesvda.py`         | cuSOLVER `gesvda` (batched tall-skinny SVD)        |
| 4    | `step4_gram_eigh.py`      | Gram matrix + `eigh`, never form U                 |
| 5    | `step5_xsyev_fp32.py`     | `cusolverDnXsyevBatched` in FP32 via nvmath-python |
| 6    | `step6_triton_scatter.py` | Triton kernel for the weighted scatter-add         |
| 7    | `step7_triton_fused.py`   | fused Triton Gram and reconstruction kernels       |
| 8    | `step8_tuned.py`          | tuned launch configuration, pinned D2H copy        |

Results on the default problem (volume 96x96x48x48, 11x11x11 patches, stride 3,
12600 patches) are in [`results/summary.org`](results/summary.org): 15.1 s for
step 0 down to 66 ms for step 8.

### Results

Volume (96, 96, 48, 48), patch 11x11x11x48, stride 3, 12600 patches. Best of 5, wall clock, host->host.

| step                                | time (ms) | us/patch | vs. step 0 | rel. err. vs step 0 |
|-------------------------------------|-----------|----------|------------|---------------------|
| 0  NumPy loop (1 BLAS thread)       | 15,107.2  | 1,198.98 | 1.0x       | 0.0e+00             |
| 1  PyTorch loop on GPU              | 14,624.9  | 1,160.71 | 1.0x       | 6.5e-07             |
| 2  batched, default SVD             | 11,746.1  | 932.23   | 1.3x       | 6.5e-07             |
| 3  batched, gesvda                  | 645.8     | 51.25    | 23x        | 1.1e-07             |
| 4  Gram + eigh, U-free (torch 2.11) | 6,450.9   | 511.98   | 2.3x       | 1.4e-07             |
| 4  Gram + eigh, U-free              | 181.8     | 14.43    | 83x        | 1.1e-07             |
| 5  XsyevBatched FP32                | 178.8     | 14.19    | 84x        | 1.1e-07             |
| 6  + Triton scatter-add             | 170.3     | 13.52    | 89x        | 1.4e-07             |
| 7  + fused Triton kernels           | 128.6     | 10.21    | 117x       | 1.5e-07             |
| 8  + tuned launch, pinned D2H       | 66.5      | 5.28     | 227x       | 1.6e-07             |


## Usage

Requires Linux, an NVIDIA GPU and [uv](https://docs.astral.sh/uv/)
(PyTorch is pulled from the CUDA 13.0 wheel index).

```sh
./run_all.sh                       # every step, then tools/summary.py
uv run python step8_tuned.py       # a single step
```

Problem size is set through environment variables: `VOL` (e.g. `96,96,48,48`),
`STEP` (stride) and `BATCH` (patches per batch, steps 2+).

- `micro/`: micro-benchmarks for the side investigations of the post (SVD
  drivers, eigensolver speed, rank agreement of the Gram route, batch size,
  patch order, kernel tuning, CPU threads).
- `tools/`: Nsight Systems profiling (`profile.sh`), kernel time breakdown,
  timeline figures and the summary table.
