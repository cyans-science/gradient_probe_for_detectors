#!/usr/bin/env python3
"""
build_event_index.py

Purpose
-------
Rebuild, after training, which event each probe `event_id` refers to.

fisher_val_loop numbers events in the order the validation loader delivers
them and keeps no other identifier. That order is fixed by djcdata:

  * the validation set is the last `val_fraction` of the .djcdc sample list
    (DataCollection.split), read unshuffled, file after file;
  * batches are filled greedily up to `batch_size` elements and run across
    file boundaries;
  * an event with more elements than `batch_size` gets a batch of its own,
    which is skipped, so it never receives an event_id.

This script replays that order from the .djcdc and the .djctd files
(read-only) and writes one row per event_id:

    event_id, djctd_file, local_idx, root_file, entry,
    n_hits, energy_sum, batch_idx, n_events_in_batch

`entry` is the root entry of the event (L3). If the conversion dropped root
entries, the .djctd events are matched to the converter's output instead.
`batch_idx` and `n_events_in_batch` are emulated from djcdata's batching
rule, not observed. `n_hits` and `energy_sum` are taken from the .djctd.

Checks
------
L1  Hits per event_id from this index == rows per event_id in a probe CSV.
    Validates the whole event_id -> (djctd_file, local_idx) mapping.
L3  local_idx -> root entry, per validation file. Needs uproot; skipped if
    unavailable.
      * same number of events in the .djctd and the root tree:
        entry = local_idx ("identity");
      * fewer .djctd events (the conversion dropped entries): with
        --converter, every root entry is converted and each .djctd event,
        in order, takes the next entry that holds at least --min-overlap of
        its hit positions ("aligned by cell positions"). Positions, not
        energies: the training conversion changed ~2% of the hit energies
        and removed a few hits, so the converter does not reproduce the
        .djctd exactly. Dropped entries and hit-count / energy agreement
        are listed in events_meta.json.
    Files that fail get entry = -1.

Outputs (in --out-dir)
----------------------
events.csv          the index
events_meta.json    inputs, versions, skipped events, check results

Nothing in djcdata, e2ereco or the .djcdc/.djctd files is modified.
If djcdata prints "wrong dataCollection format", the stored dataclass could
not be imported. That is harmless here: only the sample lists are used.

Exit code is 1 if any check failed, 0 otherwise (skipped checks included).

cyan.jo, 2026-10-21
with Claude (Anthropic)
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

INDEX_COLUMNS = [
    "event_id", "djctd_file", "local_idx", "root_file", "entry",
    "n_hits", "energy_sum", "batch_idx", "n_events_in_batch",
]
MAX_REPORTED_MISMATCHES = 10


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Map probe event_ids back to .djctd files and root entries.",
    )
    parser.add_argument("--djcdc", required=True, type=Path,
                        help="dataCollection.djcdc used for the training run")
    parser.add_argument("--root-dir", required=True, type=Path,
                        help="directory holding the source .root files")
    parser.add_argument("--out-dir", required=True, type=Path,
                        help="run directory for events.csv and events_meta.json")
    parser.add_argument("--probe-csv", type=Path, default=None,
                        help="any one stage CSV (event_id,hit_id,value), for check L1")
    parser.add_argument("--val-fraction", type=float, default=0.2,
                        help="split_fraction given to train_loader.split() (default 0.2)")
    parser.add_argument("--batch-size", type=int, default=10_000,
                        help="DJCDataLoader batch_size used in training (default 10000)")
    parser.add_argument("--feature-array", default="recHitFeatures",
                        help="ragged feature array holding one row per hit")
    parser.add_argument("--energy-column", type=int, default=0,
                        help="column of the hit energy in --feature-array (default 0)")
    parser.add_argument("--root-tree", default="Hits",
                        help="tree whose entry count is compared in check L3")
    parser.add_argument("--xyz-columns", type=int, nargs=3, default=[5, 6, 7],
                        help="columns of x, y, z in --feature-array (default 5 6 7)")
    parser.add_argument("--converter", type=Path, default=None,
                        help="file defining TrainData_DICE_cellparent; needed for L3 "
                             "when the conversion dropped root entries")
    parser.add_argument("--min-overlap", type=float, default=0.99,
                        help="fraction of .djctd hit positions an entry must hold to "
                             "match (default 0.99)")
    parser.add_argument("--root-suffix", default=".root")
    return parser.parse_args(argv)


# -----------------------------------------------------------------------------
# Reading the DJC side (read-only)
# -----------------------------------------------------------------------------
def load_validation_collection(djcdc: Path, val_fraction: float):
    """Return (validation DataCollection, number of samples before the split,
    {sample: source root file recorded at conversion}).

    Uses djcdata's own split with the same argument as DJCDataLoader.split(),
    so the validation file list and order match the training run.
    DataCollection.split() always empties the source list (it compares it
    with the already shortened sample list), so it is read before splitting.
    """
    from djcdata import DataCollection

    collection = DataCollection(str(djcdc))
    n_total = len(collection.samples)
    sources = (dict(zip(collection.samples, collection.sourceList))
               if len(collection.sourceList) == n_total else {})
    validation = collection.split(1. - val_fraction)
    return validation, n_total, sources


def read_djctd(path: Path, feature_array: str, energy_column: int,
               xyz_columns=(5, 6, 7)) -> dict:
    """Per-event counts and hit positions of one .djctd file, in stored order."""
    from djcdata import TrainData

    data = TrainData()
    data.readFromFile(str(path))
    n_events = int(data.nElements())

    # Names must be read before transfer*ToNumpy, which empties the arrays.
    named = {}
    for kind, names_of, transfer in (
        ("feature", data.getNumpyFeatureArrayNames, data.transferFeatureListToNumpy),
        ("truth", data.getNumpyTruthArrayNames, data.transferTruthListToNumpy),
        ("weight", data.getNumpyWeightArrayNames, data.transferWeightListToNumpy),
    ):
        names = list(names_of())
        named[kind] = dict(zip(names, transfer(False)))

    features = named["feature"]
    splits_name = feature_array + "_rowsplits"
    if feature_array not in features or splits_name not in features:
        raise KeyError(
            f"{path.name}: no ragged feature array {feature_array!r}. "
            f"Available: {sorted(features)}"
        )

    row_splits = np.asarray(features[splits_name]).reshape(-1).astype(np.int64)
    hits = np.asarray(features[feature_array])
    if row_splits.size != n_events + 1 or row_splits[-1] != hits.shape[0]:
        raise RuntimeError(
            f"{path.name}: row splits ({row_splits.size}, last {row_splits[-1]}) "
            f"do not match {n_events} events / {hits.shape[0]} hits."
        )

    n_hits = np.diff(row_splits)
    cumulative = np.concatenate((
        [0.0], np.cumsum(hits[:, energy_column], dtype=np.float64),
    ))
    energy_sum = cumulative[row_splits[1:]] - cumulative[row_splits[:-1]]

    # djcdata batches on the per-event sum over all *distinct* row-split
    # arrays of the file (trainDataGenerator::prepareSplitting).
    groups = []
    for arrays in named.values():
        for name, array in arrays.items():
            if not name.endswith("_rowsplits"):
                continue
            splits = np.asarray(array).reshape(-1).astype(np.int64)
            if splits.size and not any(np.array_equal(splits, g) for g in groups):
                groups.append(splits)
    if any(g.size != n_events + 1 for g in groups):
        raise RuntimeError(f"{path.name}: row-split arrays disagree on the event count.")
    n_elements = np.sum([np.diff(g) for g in groups], axis=0)

    return {
        "n_events": n_events,
        "n_hits": n_hits,
        "energy_sum": energy_sum,
        "n_elements": n_elements,
        "n_rowsplit_groups": len(groups),
        "row_splits": row_splits,
        "xyz": hits[:, list(xyz_columns)].copy(),
    }


# -----------------------------------------------------------------------------
# Replaying the validation order
# -----------------------------------------------------------------------------
def emulate_batches(n_elements, limit: int, squared_limit: bool = False,
                    skip_large: bool = True):
    """Batches of the validation stream as (first_event, stop_event, used).

    Line-by-line port of djcdata simpleArrayBase::priv_getSplitIndices,
    applied to the whole stream as trainDataGenerator does. `used` is False
    for batches the generator skips: a single event above the limit, or a
    batch without elements.
    """
    row_splits = np.concatenate(([0], np.cumsum(n_elements))).astype(np.int64)
    n_bounds = row_splits.size
    batches = []
    i_old, s_old, i_s = 0, 0, 0

    while True:
        delta = int(row_splits[i_s]) - s_old
        if squared_limit:
            delta *= delta
        split_at = None

        if delta > limit and i_s != i_old + 1:
            split_at = i_s - 1
            i_s -= 1
        elif delta == limit or i_s == n_bounds - 1 or (delta > limit and i_s == i_old + 1):
            split_at = i_s

        if split_at is not None:
            if split_at == i_old:
                raise RuntimeError(f"empty batch at event {split_at}")
            n_in_batch = int(row_splits[split_at] - row_splits[i_old])
            used = (not skip_large or n_in_batch <= limit) and n_in_batch > 0
            batches.append((i_old, split_at, used))
            i_old = split_at
            s_old = int(row_splits[i_old])

        i_s += 1
        if i_s >= n_bounds:
            break

    return batches


def build_index(files: list, batch_size: int, squared_limit: bool):
    """Assign event_ids in validation order. Returns (index, skipped, n_batches)."""
    file_of = np.concatenate([np.full(f["n_events"], i) for i, f in enumerate(files)])
    local_of = np.concatenate([np.arange(f["n_events"]) for f in files])
    n_hits = np.concatenate([f["n_hits"] for f in files])
    energy_sum = np.concatenate([f["energy_sum"] for f in files])
    n_elements = np.concatenate([f["n_elements"] for f in files])
    if n_elements.size == 0:
        raise RuntimeError("The validation files contain no events.")

    rows, skipped = [], []
    event_id = batch_idx = 0
    for start, stop, used in emulate_batches(n_elements, batch_size, squared_limit):
        if not used:
            skipped.extend({
                "djctd_file": files[file_of[k]]["djctd_file"],
                "local_idx": int(local_of[k]),
                "n_elements": int(n_elements[k]),
            } for k in range(start, stop))
            continue
        for k in range(start, stop):
            source = files[file_of[k]]
            rows.append((
                event_id, source["djctd_file"], int(local_of[k]),
                source["root_file"], int(local_of[k]),
                int(n_hits[k]), float(energy_sum[k]),
                batch_idx, stop - start,
            ))
            event_id += 1
        batch_idx += 1

    return pd.DataFrame(rows, columns=INDEX_COLUMNS), skipped, batch_idx


# -----------------------------------------------------------------------------
# Checks
# -----------------------------------------------------------------------------
def check_l1(index: pd.DataFrame, probe_csv) -> dict:
    """Hits per event_id in the index == rows per event_id in the probe CSV."""
    if probe_csv is None:
        return {"status": "skipped", "reason": "no --probe-csv given"}

    event_ids = pd.read_csv(probe_csv, usecols=["event_id"])["event_id"].to_numpy(np.int64)
    expected = index["n_hits"].to_numpy()
    result = {
        "probe_csv": str(probe_csv),
        "probe_rows": int(event_ids.size),
        "index_events": int(expected.size),
    }
    if event_ids.size == 0:
        return {**result, "status": "failed", "reason": "probe CSV has no rows"}
    if event_ids.min() < 0:
        return {**result, "status": "failed", "reason": "negative event_id in probe CSV"}

    observed = np.bincount(event_ids, minlength=expected.size)
    if observed.size > expected.size:
        return {
            **result, "status": "failed",
            "reason": (f"probe CSV reaches event_id {int(event_ids.max())}, "
                       f"the index has only {expected.size} events"),
        }

    bad = np.flatnonzero(observed != expected)
    if bad.size == 0:
        return {**result, "status": "passed"}

    return {
        **result,
        "status": "failed",
        "reason": f"{bad.size} event_ids differ in hit count",
        "first_mismatches": [
            {
                "event_id": int(i),
                "index_hits": int(expected[i]),
                "probe_rows": int(observed[i]),
                "djctd_file": index.at[i, "djctd_file"],
                "local_idx": int(index.at[i, "local_idx"]),
            }
            for i in bad[:MAX_REPORTED_MISMATCHES]
        ],
    }


def load_converter(path):
    """Instance of TrainData_DICE_cellparent, imported from its file."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("TrainData_DICE_cellparent", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TrainData_DICE_cellparent()


def position_keys(xyz) -> set:
    """Hit positions rounded to 0.1 mm, taken as float32 like the .djctd stores them."""
    rounded = np.round(np.asarray(xyz, dtype=np.float32).astype(np.float64), 1)
    return set(map(tuple, rounded))


def converted_entries(root_file: Path, converter, energy_name="recHitEnergy"):
    """(hit count, energy sum, hit positions) of every root entry, as the converter makes them.

    Entries without SimShowers give 0 hits, as in gradient_hit_viewer.py.
    Energies are summed as float32 values, like the .djctd stores them.
    """
    import uproot

    with uproot.open(root_file) as handle:
        hits = handle["Hits"].arrays(library="ak")
        showers = handle["SimShowers"].arrays(library="ak")

    out = []
    for entry in range(len(hits)):
        if entry and entry % 250 == 0:
            print(f"    {entry} / {len(hits)} entries converted")
        if len(showers["sim_shower_pdg_id"][entry]) == 0:
            out.append((0, 0.0, set()))
            continue
        converted = converter.convertevent(hits[entry], showers[entry])
        energy = (converted[energy_name].to_numpy(np.float32).sum(dtype=np.float64)
                  if len(converted) else 0.0)
        keys = position_keys(converted[["recHitX", "recHitY", "recHitZ"]].to_numpy())
        out.append((len(converted), energy, keys))
    return out


def align_entries(source: dict, converted: list, min_overlap=0.99, window=200):
    """Root entry of each .djctd event, matching in order by hit positions.

    Each .djctd event takes the earliest following entry that holds at least
    `min_overlap` of its hit positions; other events share only a few
    percent of positions with it. Returns (entries, overlaps, None), or
    (None, None, message) for the first event without a match.
    """
    row_splits, xyz = source["row_splits"], source["xyz"]
    n_events = len(row_splits) - 1
    entries = np.empty(n_events, dtype=np.int64)
    overlaps = np.empty(n_events)
    j = 0
    for k in range(n_events):
        keys = position_keys(xyz[row_splits[k]:row_splits[k + 1]])
        best_overlap, best_entry = -1.0, None
        for candidate in range(j, min(j + window, len(converted))):
            n_converted, _, converted_keys = converted[candidate]
            if keys:
                overlap = len(keys & converted_keys) / len(keys)
            else:
                overlap = 1.0 if n_converted == 0 else 0.0
            if overlap > best_overlap:
                best_overlap, best_entry = overlap, candidate
            if overlap >= min_overlap:
                break
        if best_overlap < min_overlap:
            return None, None, (
                f"djctd event {k} ({row_splits[k + 1] - row_splits[k]} hits): best "
                f"overlap {best_overlap:.3f} at entry {best_entry}, searched entries "
                f"{j}-{min(j + window, len(converted)) - 1}")
        entries[k], overlaps[k] = best_entry, best_overlap
        j = best_entry + 1
    return entries, overlaps, None


def check_l3(index: pd.DataFrame, files: list, tree: str, converter=None,
             min_overlap=0.99) -> dict:
    """Root entry of every .djctd event, per validation file."""
    try:
        import uproot
    except ImportError:
        return {"status": "skipped",
                "reason": "uproot not importable here; entry == local_idx is assumed"}

    per_file, failed_files = [], []
    for source in files:
        root_file = Path(source["root_file"])
        record = {
            "djctd_file": source["djctd_file"],
            "root_file": str(root_file),
            "djctd_events": source["n_events"],
            "root_entries": None,
        }
        per_file.append(record)

        if not root_file.is_file():
            record.update(ok=False, reason="root file not found")
            failed_files.append(source["djctd_file"])
            continue
        with uproot.open(root_file) as handle:
            record["root_entries"] = int(handle[tree].num_entries)

        if record["root_entries"] == source["n_events"]:
            record.update(ok=True, method="identity")
            continue
        if converter is None:
            record.update(ok=False, reason="the conversion dropped entries; "
                                           "pass --converter to match them")
            failed_files.append(source["djctd_file"])
            continue

        print(f"  matching {root_file.name}: converting {record['root_entries']} entries")
        converted = converted_entries(root_file, converter)
        entries, overlaps, problem = align_entries(source, converted, min_overlap)
        method = "aligned by cell positions"
        if entries is None:
            record.update(ok=False, method=method, reason=problem)
            failed_files.append(source["djctd_file"])
            continue

        rows = index["djctd_file"] == source["djctd_file"]
        index.loc[rows, "entry"] = entries[index.loc[rows, "local_idx"].to_numpy()]
        root_hits = np.array([c[0] for c in converted])
        root_energy = np.array([c[1] for c in converted])
        dropped = np.setdiff1d(np.arange(len(converted)), entries)
        record.update(
            ok=True, method=method,
            min_overlap=float(overlaps.min()) if len(overlaps) else None,
            same_hit_count=int((root_hits[entries] == source["n_hits"]).sum()),
            same_energy_sum=int(np.isclose(root_energy[entries], source["energy_sum"],
                                           rtol=1e-6, atol=1e-9).sum()),
            n_dropped=int(dropped.size),
            dropped_entries=[{"entry": int(e), "converted_hits": int(root_hits[e])}
                             for e in dropped],
        )

    if failed_files:
        index.loc[index["djctd_file"].isin(failed_files), "entry"] = -1

    return {
        "status": "failed" if failed_files else "passed",
        "tree": tree,
        "failed_files": failed_files,
        "per_file": per_file,
    }


# -----------------------------------------------------------------------------
def djcdata_version():
    try:
        from importlib.metadata import version
        return version("djcdata")
    except Exception:
        return None


def main(argv=None) -> int:
    args = parse_args(argv)
    import djcdata

    validation, n_total, recorded_sources = load_validation_collection(
        args.djcdc, args.val_fraction)
    squared_limit = bool(getattr(validation, "batch_uses_sum_of_squares", False))
    print(f"{len(validation.samples)} of {n_total} samples are validation files.")

    warnings, files = [], []
    for sample in validation.samples:
        source = recorded_sources.get(sample)
        djctd_file = Path(validation.getSamplePath(sample)).resolve()
        stem = Path(sample).stem
        root_file = args.root_dir / (stem + args.root_suffix)

        if source is not None and Path(source).stem != stem:
            warnings.append(f"{sample}: .djcdc source list names {source}")
        if not root_file.is_file():
            warnings.append(f"root file not found: {root_file}")

        print(f"  reading {djctd_file.name}")
        counts = read_djctd(djctd_file, args.feature_array, args.energy_column,
                            args.xyz_columns)
        files.append({
            "djctd_file": str(djctd_file),
            "root_file": str(root_file),
            "source": source,
            **counts,
        })

    index, skipped, n_batches = build_index(files, args.batch_size, squared_limit)
    converter = load_converter(args.converter) if args.converter is not None else None
    checks = {
        "L1": check_l1(index, args.probe_csv),
        "L3": check_l3(index, files, args.root_tree, converter, args.min_overlap),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.out_dir / "events.csv"
    meta_path = args.out_dir / "events_meta.json"
    index.to_csv(index_path, index=False)

    meta = {
        "script": Path(__file__).name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "inputs": {key: (str(value) if value is not None else None)
                   for key, value in vars(args).items()},
        "djcdata": {"module": djcdata.__file__, "version": djcdata_version()},
        "n_samples_total": n_total,
        "validation_files": [
            {"djctd_file": f["djctd_file"], "root_file": f["root_file"],
             "source": f["source"], "n_events": f["n_events"],
             "n_rowsplit_groups": f["n_rowsplit_groups"]}
            for f in files
        ],
        "squared_elements_limit": squared_limit,
        "n_stream_events": int(sum(f["n_events"] for f in files)),
        "n_event_ids": int(len(index)),
        "n_batches": n_batches,
        "n_skipped_events": len(skipped),
        "skipped_events": skipped,
        "emulated_columns": ["batch_idx", "n_events_in_batch"],
        "warnings": warnings,
        "checks": checks,
    }
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    print(f"\n{len(index)} event_ids in {n_batches} batches; "
          f"{len(skipped)} events skipped as larger than the batch size.")
    for message in warnings:
        print(f"WARNING: {message}")
    for name, check in checks.items():
        line = f"{name}: {check['status']}"
        if "reason" in check:
            line += f" ({check['reason']})"
        print(line)
        for mismatch in check.get("first_mismatches", []):
            print(f"    {mismatch}")
        for record in check.get("per_file", []):
            line = (f"    {Path(record['djctd_file']).name}: {record['djctd_events']} "
                    f"djctd events, {record['root_entries']} root entries")
            if "method" in record:
                line += f", {record['method']}"
            if "n_dropped" in record:
                n = record["djctd_events"]
                line += (f", {record['n_dropped']} entries dropped by the conversion; "
                         f"lowest overlap {record['min_overlap']:.3f}; same hit count "
                         f"{record['same_hit_count']}/{n}, same energy sum "
                         f"{record['same_energy_sum']}/{n}")
            if "reason" in record:
                line += f" -- {record['reason']}"
            print(line)
    print(f"\nWrote {index_path}\nWrote {meta_path}")

    return 1 if any(c["status"] == "failed" for c in checks.values()) else 0


if __name__ == "__main__":
    sys.exit(main())