"""Render an Nsight Systems capture as a static timeline figure.

Reads the SQLite export written by ``tools/profile.sh`` and draws the rows the
Nsight Systems GUI would show -- NVTX ranges, CUDA API calls on the host thread,
kernels and memcpys on the GPU -- for a time window, with kernels coloured by
which library they come from.

    uv run python tools/nsys_timeline.py results/prof/step1_torch_loop.sqlite \
        --start-ms 10 --dur-ms 2 -o figures/step1_timeline.png
"""

import argparse
import re
import sqlite3

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

# (label, regex on the demangled kernel name, colour) -- first match wins.
# Colours are the reference categorical palette, assigned in fixed order.
CATEGORIES = [
    ("cuSOLVER, FP64 (!)", r"double|d884|dgemm|_d_|<double", "#e34948"),
    (
        "cuSOLVER (SVD / eig)",
        r"gesv|syev|svd_|geqr|lar[fg]|laed|sytrd|steqr|orm|offA|"
        r"pegasus|stedc|sterf|lacpy|syevj|jacobi|householder|sytd2|ormtr|dsyev|cusolver",
        "#2a78d6",
    ),
    ("cuBLAS (GEMM)", r"gemm|cublas|splitK|cutlass|xmma|sm\d+_", "#eb6834"),
    ("PyTorch (elementwise / reduce / index)", r"at::native|at_cuda|cub::", "#1baf7a"),
    ("Triton (ours)", r"^_|triton|_kernel$", "#eda100"),
    ("other", r".", "#4a3aa7"),
]
MEMCPY_COLOR = "#e87ba4"
SYNC_COLOR = "#0b0b0b"
API_COLOR = "#b5b3ad"


def categorize(name):
    for label, pat, color in CATEGORIES:
        if re.search(pat, name):
            return label, color
    raise AssertionError


def load(db, start_ms, dur_ms):
    c = sqlite3.connect(db)
    names = dict(c.execute("select id, value from StringIds"))
    # Time origin: the first GPU activity of the captured call.
    t0 = c.execute(
        "select min(start) from (select start from CUPTI_ACTIVITY_KIND_KERNEL "
        "union all select start from CUPTI_ACTIVITY_KIND_RUNTIME)"
    ).fetchone()[0]
    lo = t0 + int(start_ms * 1e6)
    hi = lo + int(dur_ms * 1e6) if dur_ms else 2**62

    def q(sql):
        return [
            ((s - lo) / 1e6, (e - s) / 1e6, *rest)
            for s, e, *rest in c.execute(sql, (hi, lo))
        ]

    kernels = q(
        "select start, end, demangledName from CUPTI_ACTIVITY_KIND_KERNEL "
        "where start < ? and end > ?"
    )
    kernels = [(s, d, names[n]) for s, d, n in kernels]
    memcpy = q(
        "select start, end, bytes from CUPTI_ACTIVITY_KIND_MEMCPY "
        "where start < ? and end > ?"
    )
    api = q(
        "select start, end, nameId from CUPTI_ACTIVITY_KIND_RUNTIME "
        "where start < ? and end > ?"
    )
    api = [(s, d, names[n]) for s, d, n in api]
    nvtx = q(
        "select start, end, text from NVTX_EVENTS "
        "where end is not null and text is not null and start < ? and end > ?"
    )
    span = (hi - lo) / 1e6 if dur_ms else max(s + d for s, d, *_ in kernels + api)
    return kernels, memcpy, api, nvtx, span


def busy_fraction(intervals, span):
    """Fraction of [0, span] covered by at least one interval."""
    covered, cur_s, cur_e = 0.0, None, None
    for s, d in sorted(
        (max(s, 0), min(s + d, span) - max(s, 0)) for s, d, *_ in intervals
    ):
        if d <= 0:
            continue
        if cur_e is None or s > cur_e:
            covered += 0 if cur_e is None else cur_e - cur_s
            cur_s, cur_e = s, s + d
        else:
            cur_e = max(cur_e, s + d)
    covered += 0 if cur_e is None else cur_e - cur_s
    return covered / span


def draw(db, start_ms, dur_ms, out, title, unit="ms"):
    kernels, memcpy, api, nvtx, span = load(db, start_ms, dur_ms)
    scale = {"ms": 1.0, "us": 1e3}[unit]
    rows = ["NVTX", "CUDA API (host)", "GPU kernels", "GPU memcpy"]
    fig, ax = plt.subplots(figsize=(11, 2.9), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    h = 0.62

    def bars(y, items, color_of):
        for s, d, *rest in items:
            ax.broken_barh(
                [(s * scale, max(d * scale, span * scale / 4000))],
                (y - h / 2, h),
                facecolors=color_of(*rest),
                linewidth=0,
            )

    # nested NVTX ranges: draw outermost first, label the ones wide enough
    for s, d, text in sorted(nvtx, key=lambda r: -r[1]):
        ax.broken_barh(
            [(s * scale, d * scale)],
            (3 - h / 2, h),
            facecolors="#e9e7f6",
            edgecolors="#4a3aa7",
            linewidth=0.6,
        )
        if d / span > 0.06:
            ax.text(
                (max(s, 0) + min(d, span - max(s, 0)) / 2) * scale,
                3,
                text,
                ha="center",
                va="center",
                fontsize=7.5,
                color=INK,
                clip_on=True,
            )
    bars(
        2,
        api,
        lambda n: SYNC_COLOR if re.search("Synchronize|Memcpy", n) else API_COLOR,
    )
    bars(1, kernels, lambda n: categorize(n)[1])
    bars(0, memcpy, lambda b: MEMCPY_COLOR)

    util = busy_fraction(kernels + memcpy, span)
    ax.set_yticks(range(4)[::-1], rows, fontsize=9, color=INK)
    ax.set_xlim(0, span * scale)
    ax.set_ylim(-0.6, 3.6)
    ax.set_xlabel(f"time ({unit})", fontsize=9, color=MUTED)
    ax.tick_params(axis="x", colors=MUTED, labelsize=8)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_title(
        f"{title}   —   GPU busy {100 * util:.0f}% of window",
        fontsize=10,
        color=INK,
        loc="left",
    )

    seen = {categorize(n)[0] for *_, n in kernels}
    handles = [Patch(color=c, label=lbl) for lbl, _, c in CATEGORIES if lbl in seen]
    handles += [
        Patch(color=SYNC_COLOR, label="host waits on GPU (synchronize / memcpy)"),
        Patch(color=API_COLOR, label="other CUDA API (kernel launch, ...)"),
    ]
    if memcpy:
        handles.append(Patch(color=MEMCPY_COLOR, label="memcpy"))
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.32),
        ncol=4,
        fontsize=7.5,
        frameon=False,
        labelcolor=INK,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(f"wrote {out}  (window {span:.3f} ms, GPU busy {100 * util:.1f}%)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("--start-ms", type=float, default=0.0)
    ap.add_argument("--dur-ms", type=float, default=0.0, help="0: whole capture")
    ap.add_argument("--unit", choices=["ms", "us"], default="ms")
    ap.add_argument("--title", default="")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    draw(a.db, a.start_ms, a.dur_ms, a.out, a.title, a.unit)
