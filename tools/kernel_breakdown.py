"""GPU time per kernel category for one or more profiles, as an Org table.

uv run python tools/kernel_breakdown.py results/prof/step5_xsyev_fp32.sqlite ...
"""

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nsys_timeline import CATEGORIES, categorize  # noqa: E402


def breakdown(db):
    c = sqlite3.connect(db)
    names = dict(c.execute("select id, value from StringIds"))
    tot = {}
    for start, end, n in c.execute(
        "select start, end, demangledName from CUPTI_ACTIVITY_KIND_KERNEL"
    ):
        label = categorize(names[n])[0]
        tot[label] = tot.get(label, 0.0) + (end - start) / 1e6
    for kind, label in ((1, "memcpy H2D"), (2, "memcpy D2H")):
        (ms,) = c.execute(
            "select coalesce(sum(end - start), 0) / 1e6 from CUPTI_ACTIVITY_KIND_MEMCPY "
            "where copyKind = ?",
            (kind,),
        ).fetchone()
        tot[label] = ms
    return tot


if __name__ == "__main__":
    dbs = sys.argv[1:]
    data = {Path(db).stem: breakdown(db) for db in dbs}
    labels = [lbl for lbl, *_ in CATEGORIES] + ["memcpy H2D", "memcpy D2H"]
    labels = [lbl for lbl in labels if any(d.get(lbl, 0) > 0.05 for d in data.values())]
    print("| GPU time (ms) | " + " | ".join(data) + " |")
    print("|---" * (len(data) + 1) + "|")
    for lbl in labels:
        print(
            f"| {lbl} | "
            + " | ".join(f"{d.get(lbl, 0):.1f}" for d in data.values())
            + " |"
        )
    print(
        "| *total* | "
        + " | ".join(f"*{sum(d.values()):.1f}*" for d in data.values())
        + " |"
    )
