#!/usr/bin/env bash
# Run every rung on the default problem and write results/*.json, then the summary.
#   ./run_all.sh
# Set OLD_TORCH_PY to a python with an older PyTorch (e.g. 2.11) to also record
# step 4 there, where batched eigh still uses Jacobi.
set -euo pipefail
cd "$(dirname "$0")"
OPENBLAS_NUM_THREADS=1 uv run python step0_numpy_loop.py
for s in step1_torch_loop step2_torch_batched step3_gesvda step4_gram_eigh \
         step5_xsyev_fp32 step6_triton_scatter step7_triton_fused step8_tuned; do
    uv run python "$s.py"
done
if [[ -n "${OLD_TORCH_PY:-}" ]]; then
    SUFFIX=_torch2.11 "$OLD_TORCH_PY" step4_gram_eigh.py
fi
uv run python tools/summary.py
