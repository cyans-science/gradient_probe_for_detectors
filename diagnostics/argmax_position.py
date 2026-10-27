"""
argmax_position.py

Purpose: check where in an event the strongest gradient sits, using only the
probe CSVs (event_id, hit_id, value). If the hit with the largest |g| were
almost always the last hit_id (or the first), that would point to the hit
order or the event boundaries rather than to physics.

Per stage, for the hit with the largest |g|, the largest g and the smallest g:
  at_last   fraction of events where it is hit_id n-1
  at_first  fraction of events where it is hit_id 0
  uniform   what at_last / at_first would be if the position were random (mean 1/n)
  median r  median of hit_id / (n-1)  (0.5 if random)

Run:
  python3 diagnostics/argmax_position.py --probe-dir <dir with <stage>.csv>

cyan.jo, Claude Opus 5.5, 2026-10-23
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def per_event_positions(path: Path) -> pd.DataFrame:
    """One row per event (with at least 2 hits): n and the hit_ids of the extremes."""
    df = pd.read_csv(path, dtype={"event_id": "int32", "hit_id": "int32", "value": "float32"})
    grouped = df.groupby("event_id", sort=False)
    out = pd.DataFrame({
        "n": grouped.size(),
        "abs": df.loc[df["value"].abs().groupby(df["event_id"]).idxmax(), ["event_id", "hit_id"]]
                 .set_index("event_id")["hit_id"],
        "max": df.loc[grouped["value"].idxmax(), ["event_id", "hit_id"]].set_index("event_id")["hit_id"],
        "min": df.loc[grouped["value"].idxmin(), ["event_id", "hit_id"]].set_index("event_id")["hit_id"],
    })
    return out[out["n"] >= 2]


def summarise(events: pd.DataFrame) -> pd.DataFrame:
    uniform = float((1.0 / events["n"]).mean())
    rows = {}
    for which, label in (("abs", "largest |g|"), ("max", "largest g"), ("min", "smallest g")):
        position = events[which]
        rows[label] = {
            "at_last": float((position == events["n"] - 1).mean()),
            "at_first": float((position == 0).mean()),
            "uniform": uniform,
            "median r": float(np.median(position / (events["n"] - 1))),
        }
    return pd.DataFrame(rows).T


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--probe-dir", required=True, type=Path)
    parser.add_argument("--examples", type=int, default=5,
                        help="events listed where the largest |g| is the last hit (default 5)")
    args = parser.parse_args(argv)

    files = sorted(args.probe_dir.glob("*.csv"))
    if not files:
        raise SystemExit(f"no CSV files in {args.probe_dir}")
    pd.set_option("display.float_format", "{:.4f}".format)

    for path in files:
        events = per_event_positions(path)
        print(f"\n=== {path.stem}: {len(events):,} events with >= 2 hits")
        print(summarise(events).to_string())
        last = events[events["abs"] == events["n"] - 1]
        if len(last) and args.examples:
            print(f"  e.g. events where the largest |g| is the last hit: "
                  f"{last.index[:args.examples].tolist()} (n = {last['n'].iloc[:args.examples].tolist()})")


if __name__ == "__main__":
    main()