"""
compare_djctd_root.py

Purpose
-------
Diagnostic for check L3 when the matching fails. For the first events of one
validation .djctd file, find the root entry whose converted hits sit in the
same cells, and compare the two hit by hit.

For each .djctd event it prints
  * the best root entry and the runner-up, with the fraction of .djctd hit
    positions found in that entry (positions rounded to 0.1 mm);
  * the hit counts on both sides;
  * for the hits found on both sides: the energy ratio djctd / converter
    and the fraction of hits with the same energy.
For the first event it also lists the first hits side by side.

How to read it
  * overlap ~1, ratio 1:        same hits, L3 matching should have worked
  * overlap ~1, ratio != 1:     same cells, energies changed at conversion
                                (smearing or calibration)
  * overlap < 1, counts differ: hits added or removed at conversion
  * overlap ~0 everywhere:      other converter version or other root files

Usage (inside the plotting container):
  python3 diagnostics/compare_djctd_root.py \
      --run-dir /ceph/cjo/GradientProbe/runs/20261019_Fisher_depth_probe \
      --converter <path to TrainData_DICE_cellparent.py> --file singlejet_81

cyan.jo, Claude Opus 5.5, 2026-10-21
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--run-dir", required=True, type=Path,
                        help="run directory with events_meta.json")
    parser.add_argument("--converter", required=True, type=Path,
                        help="file defining TrainData_DICE_cellparent")
    parser.add_argument("--file", default=None,
                        help="validation file stem, e.g. singlejet_81 (default: the first)")
    parser.add_argument("--n-events", type=int, default=5,
                        help=".djctd events to compare (default 5)")
    parser.add_argument("--window", type=int, default=40,
                        help="root entries searched after the previous match (default 40)")
    parser.add_argument("--feature-array", default="recHitFeatures")
    parser.add_argument("--energy-column", type=int, default=0)
    parser.add_argument("--xyz-columns", type=int, nargs=3, default=[5, 6, 7])
    return parser.parse_args(argv)


def load_converter(path):
    spec = importlib.util.spec_from_file_location("TrainData_DICE_cellparent", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TrainData_DICE_cellparent()


def read_djctd_events(path, feature_array, energy_column, xyz_columns):
    """[(xyz (n, 3), energy (n,)) per event] as stored in the .djctd."""
    from djcdata import TrainData

    data = TrainData()
    data.readFromFile(str(path))
    names = list(data.getNumpyFeatureArrayNames())
    arrays = dict(zip(names, data.transferFeatureListToNumpy(False)))
    hits = np.asarray(arrays[feature_array])
    splits = np.asarray(arrays[feature_array + "_rowsplits"]).reshape(-1)
    return [
        (hits[a:b][:, xyz_columns].astype(np.float64), hits[a:b, energy_column].astype(np.float64))
        for a, b in zip(splits[:-1], splits[1:])
    ]


class ConvertedEntries:
    """Converter output per root entry, converted on first use."""

    def __init__(self, root_file, converter):
        import uproot

        with uproot.open(root_file) as handle:
            self.hits = handle["Hits"].arrays(library="ak")
            self.showers = handle["SimShowers"].arrays(library="ak")
        self.converter = converter
        self.cache = {}

    def __len__(self):
        return len(self.hits)

    def __getitem__(self, entry):
        if entry not in self.cache:
            if len(self.showers["sim_shower_pdg_id"][entry]) == 0:
                xyz, energy = np.zeros((0, 3)), np.zeros(0)
            else:
                df = self.converter.convertevent(self.hits[entry], self.showers[entry])
                # float32 first, as stored in the .djctd, so rounding agrees.
                xyz = df[["recHitX", "recHitY", "recHitZ"]].to_numpy(np.float32).astype(np.float64)
                energy = df["recHitEnergy"].to_numpy(np.float64)
            self.cache[entry] = (xyz, energy)
        return self.cache[entry]


def position_keys(xyz):
    return [tuple(row) for row in np.round(xyz, 1)]


def compare(djctd_event, entry_event):
    """(overlap fraction, indices into djctd, indices into entry) of shared cells."""
    d_keys = position_keys(djctd_event[0])
    e_index = {key: i for i, key in enumerate(position_keys(entry_event[0]))}
    pairs = [(i, e_index[key]) for i, key in enumerate(d_keys) if key in e_index]
    overlap = len(pairs) / max(len(d_keys), 1)
    if not pairs:
        return overlap, np.zeros(0, int), np.zeros(0, int)
    d_idx, e_idx = map(np.array, zip(*pairs))
    return overlap, d_idx, e_idx


def main(argv=None):
    args = parse_args(argv)
    with open(args.run_dir / "events_meta.json", encoding="utf-8") as handle:
        files = json.load(handle)["validation_files"]
    if args.file is not None:
        files = [f for f in files if Path(f["djctd_file"]).stem == args.file]
        if not files:
            raise SystemExit(f"{args.file} is not a validation file of this run")
    source = files[0]
    print(f"{Path(source['djctd_file']).name}  vs  {source['root_file']}")

    events = read_djctd_events(source["djctd_file"], args.feature_array,
                               args.energy_column, args.xyz_columns)
    entries = ConvertedEntries(source["root_file"], load_converter(args.converter))
    print(f"{len(events)} djctd events, {len(entries)} root entries\n")

    start = 0
    for k in range(min(args.n_events, len(events))):
        candidates = range(start, min(start + args.window, len(entries)))
        scores = sorted(((compare(events[k], entries[j])[0], j) for j in candidates),
                        reverse=True)
        (best_overlap, best), runner_up = scores[0], (scores[1] if len(scores) > 1 else (0, -1))
        overlap, d_idx, e_idx = compare(events[k], entries[best])
        d_energy, e_energy = events[k][1][d_idx], entries[best][1][e_idx]
        line = (f"djctd {k:3d}: best entry {best:4d} overlap {overlap:5.3f} "
                f"(next: entry {runner_up[1]} {runner_up[0]:5.3f}) | hits djctd "
                f"{len(events[k][1])}, converter {len(entries[best][1])}")
        if len(d_idx):
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = d_energy / e_energy
            same = np.isclose(d_energy, e_energy, rtol=1e-6, atol=0)
            line += (f" | E ratio median {np.nanmedian(ratio):.6f} "
                     f"[{np.nanmin(ratio):.4f}, {np.nanmax(ratio):.4f}], "
                     f"same E {same.mean():.3f}")
        print(line)

        if k == 0 and len(d_idx):
            print("   first shared hits (x, y, z; E djctd, E converter):")
            for i, j in list(zip(d_idx, e_idx))[:8]:
                x, y, z = events[k][0][i]
                print(f"     {x:9.1f} {y:9.1f} {z:9.1f}   {events[k][1][i]:12.6g} "
                      f"{entries[best][1][j]:12.6g}   djctd hit {i}, converter hit {j}")
        if best_overlap > 0.5:
            start = best + 1


if __name__ == "__main__":
    main()