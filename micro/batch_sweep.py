"""Micro-benchmark: end-to-end time vs batch size, for a few rungs.

uv run python micro/batch_sweep.py      # writes results/batch_sweep.json
"""

import json
import os
import subprocess
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
STEPS = ["step3_gesvda", "step5_xsyev_fp32", "step8_tuned"]
BATCHES = [64, 128, 256, 512, 1024, 2048, 4096]

SNIPPET = """
import time, common, {step} as s
noisy, _ = common.make_phantom()
s.denoise(noisy)
ts = []
for _ in range(5):
    t0 = time.perf_counter(); s.denoise(noisy); ts.append(time.perf_counter() - t0)
print(min(ts))
"""

res = {}
for step in STEPS:
    for b in BATCHES:
        env = dict(os.environ, BATCH=str(b))
        cmd = [sys.executable, "-c", SNIPPET.format(step=step)]
        p = subprocess.run(cmd, env=env, cwd=ROOT, capture_output=True, text=True)
        sec = float(p.stdout.split()[-1]) if p.returncode == 0 else None
        res.setdefault(step, {})[b] = sec
        print(
            f"{step:20s} BATCH={b:5d}  "
            f"{'OOM/fail' if sec is None else f'{sec * 1e3:8.1f} ms'}",
            flush=True,
        )
with open(os.path.join(ROOT, "results", "batch_sweep.json"), "w") as f:
    json.dump(res, f, indent=1)
