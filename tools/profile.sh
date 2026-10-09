#!/usr/bin/env bash
# Profile one warm call of a ladder step with Nsight Systems, export to SQLite.
#   tools/profile.sh step3_torch_batched
set -euo pipefail
step=${1:?usage: profile.sh <step_module>}
out=results/prof/${step}
NSYS=1 nsys profile --force-overwrite=true -o "$out" \
    --trace=cuda,nvtx,osrt --capture-range=cudaProfilerApi \
    --capture-range-end=stop --cuda-memory-usage=false \
    uv run python "${step}.py"
nsys export --force-overwrite=true --type sqlite -o "${out}.sqlite" "${out}.nsys-rep" >/dev/null
nsys stats --force-export=false -q --report cuda_gpu_kern_sum --format csv \
    -o "$out" "${out}.sqlite" >/dev/null
echo "wrote ${out}.sqlite and ${out}_cuda_gpu_kern_sum.csv"
