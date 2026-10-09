"""Collect results/*.json into an Org table and the ladder bar chart.

uv run python tools/summary.py
"""

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parents[1]))
from common import RESULTS, STEP, VOL_SHAPE  # noqa: E402

SURFACE, INK, MUTED, GRID, BAR = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"
FIG = Path(__file__).parents[1] / "figures"

LABELS = {
    "step0_numpy_loop": "0  NumPy loop (1 BLAS thread)",
    "step1_torch_loop": "1  PyTorch loop on GPU",
    "step2_torch_batched": "2  batched, default SVD",
    "step3_gesvda": "3  batched, gesvda",
    "step4_gram_eigh_torch2.11": "4  Gram + eigh, U-free (torch 2.11)",
    "step4_gram_eigh": "4  Gram + eigh, U-free",
    "step5_xsyev_fp32": "5  XsyevBatched FP32",
    "step6_triton_scatter": "6  + Triton scatter-add",
    "step7_triton_fused": "7  + fused Triton kernels",
    "step8_tuned": "8  + tuned launch, pinned D2H",
}


def speedup(x):
    return f"{x:.1f}x" if x < 10 else f"{x:,.0f}x"


def main():
    tag = "x".join(map(str, VOL_SHAPE)) + f"_s{STEP}"
    rows = []
    for name in LABELS:
        f = RESULTS / f"{name}_{tag}.json"
        if f.exists():
            rows.append(json.loads(f.read_text()))
    base = rows[0]["seconds"]

    print(
        f"#+caption: Volume {VOL_SHAPE}, patch 11x11x11x{VOL_SHAPE[-1]}, stride {STEP}, "
        f"{rows[0]['n_patches']} patches. Best of 5, wall clock, host->host."
    )
    print("| step | time (ms) | us/patch | vs. step 0 | rel. err. vs step 0 | RMSE |")
    print("|------+-----------+----------+------------+---------------------+------|")
    for r in rows:
        print(
            f"| {LABELS[r['name']]} | {r['seconds'] * 1e3:,.1f} | {r['us_per_patch']:,.2f} "
            f"| {speedup(base / r['seconds'])} | {r['rel_err_vs_step0']:.1e} "
            f"| {r['rmse_denoised']:.3f} |"
        )

    fig, ax = plt.subplots(figsize=(9, 0.42 * len(rows) + 0.9), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    y = list(range(len(rows)))[::-1]
    vals = [r["us_per_patch"] for r in rows]
    inexact = [r["rel_err_vs_step0"] > 1e-5 for r in rows]
    ax.barh(y, vals, height=0.62, color=BAR, edgecolor=SURFACE, linewidth=2,
            hatch=["////" if bad else "" for bad in inexact])  # fmt: skip
    for yi, v, r in zip(y, vals, rows):
        ax.text(v * 1.12, yi, f"{v:,.1f} us  ({speedup(base / r['seconds'])})",
                va="center", fontsize=8.5, color=INK)  # fmt: skip
    ax.set_yticks(y, [LABELS[r["name"]] for r in rows], fontsize=9, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(1, max(vals) * 12)
    ax.set_xlabel(
        "time per patch (us, log scale) — lower is better", fontsize=9, color=MUTED
    )
    ax.tick_params(axis="x", colors=MUTED, labelsize=8)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.6, which="both")
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_title(
        "The ladder: end-to-end time per patch (hatched: differs from step 0 "
        "beyond round-off)",
        fontsize=10,
        color=INK,
        loc="right",
    )
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    out = FIG / "ladder.png"
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(f"\nwrote {os.path.relpath(out)}", file=sys.stderr)


if __name__ == "__main__":
    main()
