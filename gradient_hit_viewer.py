"""
gradient_hit_viewer.py

Purpose
-------
3D views of single events with every hit coloured by its probe gradient.

  * Events are chosen by event_id (position in the validation pass) or taken
    from a selection written by gradient_statistics.py: per stage (layer),
    the --events-per-stage events (default 10) with the largest |z| at that
    stage, ranked within that stage only (|z| and |g| give the same order).
    Use a selection made with --min-stages 1, so every stage has its own
    top events.
  * Every picked event gets one rotation video per stage (6 for 6 stages),
    all with the same camera and zoom box, so the stages can be compared.
    They are stored once, in the event's folder; each stage folder holds
    links named by rank to the folders of its top events.
  * events.csv from build_event_index.py (check L1 passed) gives the .djctd
    event and the root entry of every event_id.
  * The hits drawn are the .djctd hits, exactly what the model got: row k
    is hit_id k. Gradients of every stage are joined on hit_id. This
    assumes the model's Preprocessing keeps the hit order (L4).
  * The root entry, converted with the unchanged converter, supplies the
    truth the .djctd lacks (t_idx, t_pid), matched by hit position.
    The converter does not reproduce the .djctd exactly (the training
    conversion changed ~2% of hit energies and removed a few hits), so
    energies are never taken from it.
  * Check L5: at least 99% of the .djctd hit positions must be in the root
    entry. Events failing L5 are not drawn.
  * Gradients come from the probe CSVs (event_id, hit_id, value) of every
    stage, joined on hit_id. z = sign(g) log10(1 + |g|/scale) is the
    transform of the histograms, with the same scale (selection_meta.json).
  * Colour = z, one symmetric scale for all events and stages; size = hit
    energy; shape = true PID. Two black rings mark, at the stage shown,
    the hit with the largest positive z (circle) and the hit with the most
    negative z (square). They are computed from the gradients of the event
    itself, not taken from the selection; in the HTML they follow the menu.

Outputs in <run-dir>/
  stages/picked_by_stage.csv   stage, rank, event_id, hit_id, g, z, event folder
  stages/<i>_<stage>/          one folder per stage, i = position in the network
    <rank>_<event> -> ../../events/<event>   link; rank 01 = largest |z| there
  events/<root stem>_e<entry>/ one folder per event
    <i>_<stage>__<view>.webp   rotating view coloured by that stage (one turn
    <i>_<stage>__<view>.mp4    at 10 deg/s). webp for VS Code, mp4 (H.264) for
                               Keynote / PowerPoint; gif via --formats. A format
                               missing next to an existing webp or gif is made
                               from that file's frames, without rendering again.
    event_info.json            identity, check L5, picks, ringed hits per stage
    annotated_hits.csv         .djctd hits + truth + g_<stage> (raw) and
                               z_<stage> (transformed, the colour) per stage
    interactive_<view>.html    menu: colour by any stage or by true object

cyan.jo, 2026-10-21 (restructured from the earlier event viewer)
with Claude (Anthropic)
2026-10-23: rings = largest +z and most negative z per stage, from the
event's gradients (before: the selection's hits).
2026-10-23: events picked per stage (top 10 by |z|); each picked event gets
all 6 stage videos in its folder; stage folders link to their ranked events.
2026-10-26: mp4 by default (Keynote); missing formats made from existing videos.
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

import uproot
from particle import Particle

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation
from PIL import Image, ImageSequence

# Utils (PlotUtils, ExternalFunctions) is put on PYTHONPATH by run_gradient_hit_viewer.sh.
try:
    from PlotUtils import setMplParam
except ModuleNotFoundError as error:
    raise SystemExit(
        f"{error}. PlotUtils lives in /work/cjo/Utils; run this script through "
        "run_gradient_hit_viewer.sh, which puts that folder on PYTHONPATH."
    ) from error
# Shared with the statistics step: same stage names and the same transform.
from gradient_statistics import STAGE_LABELS, discover_stages, latex_scientific, signed_log10

DEFAULT_CONVERTER = Path(
    "/work/cjo/e2ereco_bernoulli_reproduction/source/"
    "modules/datastructures/TrainData_DICE_cellparent.py"
)

# Reconstructed DICE v1.4.0 geometry; provisional until confirmed.
DETECTOR = {
    "barrel_inner_radius_cm": 150.0,  # Provisional: verify physical radius.
    "barrel_half_length_cm": 300.0,
    "endcap_front_z_cm": 300.0,       # Positive |z| at the ECAL entrance.
    "endcap_inner_radius_cm": 30.0,
    "endcap_outer_radius_cm": 150.0,
    "coordinate_units_per_cm": 10,    # 1 for cm; 10 for mm.
}
COORDINATE_UNIT = "mm"  # hit coordinates; must agree with coordinate_units_per_cm

# Camera and region per view. zoom_region "auto" = dense core of the event.
VIEWS = {
    "detector": dict(detector=DETECTOR, view="detector",
                     elev=10, azim=0, vertical_axis="y"),
    "zoom": dict(detector=DETECTOR, view="detector",
                 elev=5, azim=0, vertical_axis="y", zoom_region="auto"),
}

# Diverging map: red = negative z, pale yellow = 0, blue = positive z.
GRADIENT_CMAP = "Spectral"
XYZ = ["recHitX", "recHitY", "recHitZ"]

# The two ringed hits per stage. Same colour, different shape, so they stay
# visible on any fill colour of the map.
RINGS = {
    "+": dict(label="largest +z", tex=r"largest $+z$",
              mpl_marker="o", plotly_symbol="circle-open"),
    "-": dict(label="largest −z", tex=r"largest $-z$",
              mpl_marker="s", plotly_symbol="square-open"),
}


# =============================================================================
# Data access
# =============================================================================
def load_converter(path):
    """Instance of TrainData_DICE_cellparent, imported from its file."""
    spec = importlib.util.spec_from_file_location("TrainData_DICE_cellparent", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TrainData_DICE_cellparent()


def load_converted_event(root_file, entry, converter) -> pd.DataFrame:
    """Converted hits of one root entry, row order as produced by the converter."""
    with uproot.open(root_file) as f:
        hits = f["Hits"].arrays(entry_start=entry, entry_stop=entry + 1, library="ak")
        showers = f["SimShowers"].arrays(entry_start=entry, entry_stop=entry + 1, library="ak")

    if len(hits) != 1:
        raise IndexError(f"{root_file} has no entry {entry}")
    if len(showers["sim_shower_pdg_id"][0]) == 0:
        return pd.DataFrame()  # no SimShowers -> 0 converted rows
    return converter.convertevent(hits[0], showers[0]).reset_index(drop=True)


def load_converted_events(root_file, converter, max_events=None):
    """All entries of one root file (kept for notebook use)."""
    with uproot.open(root_file) as f:
        hits = f["Hits"].arrays(
            entry_stop=max_events,
            library="ak",
        )

        showers = f["SimShowers"].arrays(
            entry_stop=max_events,
            library="ak",
        )

    # Raw shower-level truth used by the converter
    truth = {
        "pid": showers["sim_shower_pdg_id"],
        "energy": showers["sim_shower_energy"],
        "posBoundary_x": showers["sim_shower_posBoundary_x"],
        "posBoundary_y": showers["sim_shower_posBoundary_y"],
        "posBoundary_z": showers["sim_shower_posBoundary_z"],
        "posBoundary_eta": showers["sim_shower_posBoundary_eta"],
        "posBoundary_phi": showers["sim_shower_posBoundary_phi"],
    }

    converted_events = []
    event_lengths = []

    for entry in range(len(hits)):

        if len(truth["pid"][entry]) == 0:
            print(
                f"Entry {entry}: "
                "no SimShowers -> 0 converted rows"
            )

            event_df = pd.DataFrame()

        else:
            event_df = converter.convertevent(
                hits[entry],
                showers[entry],
            )

        converted_events.append(event_df)
        event_lengths.append(len(event_df))

    row_splits = np.concatenate([
        [0],
        np.cumsum(event_lengths),
    ])

    df = pd.concat(
        converted_events,
        ignore_index=True,
    )

    return df, row_splits, truth


def read_event_index(run_dir: Path) -> pd.DataFrame:
    """events.csv of the run, indexed by event_id. Requires check L1 passed."""
    meta_path = run_dir / "events_meta.json"
    with open(meta_path, encoding="utf-8") as handle:
        status = json.load(handle)["checks"]["L1"]["status"]
    if status != "passed":
        raise RuntimeError(f"Check L1 is {status!r} in {meta_path}; run build_event_index.py first.")
    return pd.read_csv(run_dir / "events.csv").set_index("event_id", drop=False)


def find_selection(run_dir: Path, name=None):
    """(selection DataFrame, selection_meta) of the run, or (None, None)."""
    root = run_dir / "selections"
    if name is not None:
        directory = root / name
        if not (directory / "selection.csv").is_file():
            raise FileNotFoundError(f"No selection.csv in {directory}")
    else:
        found = sorted(p for p in root.glob("*") if (p / "selection.csv").is_file())
        if not found:
            return None, None
        if len(found) > 1:
            raise ValueError(f"Several selections in {root}; choose one with --selection: "
                             f"{[p.name for p in found]}")
        directory = found[0]

    with open(directory / "selection_meta.json", encoding="utf-8") as handle:
        meta = json.load(handle)
    print(f"Selection: {directory.name}")
    return pd.read_csv(directory / "selection.csv"), meta


def read_event_gradients(stage_files: dict, event_ids, chunksize=2_000_000) -> dict:
    """event_id -> DataFrame(hit_id, <stage>, ...) for the requested events.

    The probe CSVs are sorted by event_id, so each file is read only up to
    the last requested event.
    """
    wanted = {int(e) for e in event_ids}
    last = max(wanted)
    per_stage = {}
    for stage, path in stage_files.items():
        print(f"  reading gradients from {Path(path).name}")
        parts = []
        for chunk in pd.read_csv(
            path, chunksize=chunksize,
            dtype={"event_id": "int32", "hit_id": "int32", "value": "float32"},
        ):
            parts.append(chunk[chunk["event_id"].isin(wanted)])
            if chunk["event_id"].iloc[-1] > last:
                break
        per_stage[stage] = pd.concat(parts, ignore_index=True)

    gradients = {}
    for event_id in wanted:
        frames = {s: df[df["event_id"] == event_id] for s, df in per_stage.items()}
        table = None
        for stage, frame in frames.items():
            if table is None:
                table = frame[["hit_id"]].reset_index(drop=True)
            elif not np.array_equal(frame["hit_id"].to_numpy(), table["hit_id"].to_numpy()):
                raise ValueError(f"event {event_id}: hit_ids differ between stages")
            table[stage] = frame["value"].to_numpy()
        gradients[event_id] = table
    return gradients


# =============================================================================
# Check L5 and annotation
# =============================================================================
# Where the hits sit in the .djctd: one row per hit in this feature array.
DJCTD_FEATURE_ARRAY = "recHitFeatures"
DJCTD_COLUMNS = {"recHitEnergy": 0, "recHitX": 5, "recHitY": 6, "recHitZ": 7}
TRUTH_COLUMNS = {"t_idx": -1, "t_pid": 0}   # name -> value for hits without truth
MIN_OVERLAP = 0.99

_DJCTD_CACHE = {}


def _djctd_arrays(djctd_file) -> dict:
    """All feature and truth arrays of one .djctd file, by name (cached)."""
    from djcdata import TrainData

    if djctd_file not in _DJCTD_CACHE:
        data = TrainData()
        data.readFromFile(str(djctd_file))
        arrays = {}
        for names_of, transfer in (
            (data.getNumpyFeatureArrayNames, data.transferFeatureListToNumpy),
            (data.getNumpyTruthArrayNames, data.transferTruthListToNumpy),
        ):
            for name, array in zip(list(names_of()), transfer(False)):
                arrays.setdefault(name, np.asarray(array))
        _DJCTD_CACHE[djctd_file] = arrays
    return _DJCTD_CACHE[djctd_file]


def load_djctd_event(djctd_file, local_idx) -> pd.DataFrame:
    """Hits of one event exactly as the model got them; row k is hit_id k.

    Per-hit truth columns (t_idx, t_pid) are included if the .djctd has them.
    """
    arrays = _djctd_arrays(djctd_file)
    row_splits = arrays[DJCTD_FEATURE_ARRAY + "_rowsplits"].reshape(-1)
    start, stop = int(row_splits[local_idx]), int(row_splits[local_idx + 1])
    hits = arrays[DJCTD_FEATURE_ARRAY][start:stop]

    event = pd.DataFrame({name: hits[:, column].astype(np.float64)
                          for name, column in DJCTD_COLUMNS.items()})
    for name in TRUTH_COLUMNS:
        splits = arrays.get(name + "_rowsplits")
        if name in arrays and splits is not None and np.array_equal(splits.reshape(-1), row_splits):
            event[name] = arrays[name][start:stop].reshape(stop - start, -1)[:, 0].astype(int)
    return event


def position_keys(xyz) -> list:
    """Hit positions rounded to 0.1 mm, taken as float32 like the .djctd stores them."""
    rounded = np.round(np.asarray(xyz, dtype=np.float32).astype(np.float64), 1)
    return list(map(tuple, rounded))


def attach_truth(event: pd.DataFrame, converted: pd.DataFrame):
    """Fill truth missing from the .djctd from the converted root entry; check L5.

    Hits are matched by position. Returns (event with t_idx and t_pid, L5 result).
    """
    first_row = {}
    for i, key in enumerate(position_keys(converted[XYZ].to_numpy()) if len(converted) else []):
        first_row.setdefault(key, i)
    match = np.array([first_row.get(key, -1) for key in position_keys(event[XYZ].to_numpy())],
                     dtype=int)
    found = match >= 0
    overlap = float(found.mean()) if len(match) else 1.0

    result = {
        "djctd_hits": len(event),
        "converter_hits": len(converted),
        "hits_found_in_root_entry": int(found.sum()),
        "overlap": overlap,
        "duplicate_positions_in_converter": int(len(converted) - len(first_row)),
    }
    if found.any():
        result["same_energy"] = float(np.isclose(
            event["recHitEnergy"].to_numpy()[found],
            converted["recHitEnergy"].to_numpy()[match[found]], rtol=1e-6, atol=0).mean())

    event = event.copy()
    for name, missing in TRUTH_COLUMNS.items():
        if name in event:
            result[f"{name}_from"] = "djctd"
            if found.any() and name in converted:
                result[f"{name}_agrees_with_converter"] = float(
                    (event[name].to_numpy()[found] == converted[name].to_numpy()[match[found]]).mean())
            continue
        values = np.full(len(event), missing, dtype=int)
        if found.any():
            values[found] = converted[name].to_numpy()[match[found]]
        event[name] = values
        result[f"{name}_from"] = "converter, by hit position"

    result["status"] = "passed" if overlap >= MIN_OVERLAP else "failed"
    if result["status"] == "failed":
        result["reason"] = (f"only {overlap:.3f} of the .djctd hit positions are in the "
                            f"root entry (need {MIN_OVERLAP})")
    return event, result


def annotate_hits(event_df, gradients, stages, log_scale) -> pd.DataFrame:
    """Event hits + raw gradient g_<stage> and transformed z_<stage> per hit."""
    if len(gradients) != len(event_df):
        raise ValueError(f"{len(gradients)} gradient rows for {len(event_df)} hits")
    if not np.array_equal(gradients["hit_id"].to_numpy(), np.arange(len(event_df))):
        raise ValueError("gradient hit_ids are not 0..n-1")

    hits = event_df.reset_index(drop=True).copy()
    hits.insert(0, "hit_id", np.arange(len(hits)))
    for stage in stages:
        hits[f"g_{stage}"] = gradients[stage].to_numpy()
        hits[f"z_{stage}"] = signed_log10(hits[f"g_{stage}"], scale=log_scale)
    return hits


def extreme_hits(hits, stage) -> dict:
    """{"+": hit_id of the largest positive z, "-": hit_id of the most negative z}.

    z grows with g, so these are also the hits with the largest positive and
    the most negative raw gradient. A sign without any hit is left out.
    """
    z = hits[f"z_{stage}"]
    extremes = {}
    if (z > 0).any():
        extremes["+"] = int(hits.loc[z.idxmax(), "hit_id"])
    if (z < 0).any():
        extremes["-"] = int(hits.loc[z.idxmin(), "hit_id"])
    return extremes


# =============================================================================
# Scene helpers shared by both renderers
# =============================================================================
def auto_zoom_region(hits, quantile=0.9, pad=0.15, focus_hit_ids=()):
    """Box around the dense core of an event.

    Per axis: the energy-weighted central `quantile` of hit positions,
    stretched to include the focus hits and widened by `pad` of its span.
    """
    if hits.empty:
        return None
    xyz = hits[XYZ].to_numpy(dtype=float)
    weights = hits["recHitEnergy"].clip(lower=0).to_numpy(dtype=float)
    if weights.sum() <= 0:
        weights = np.ones(len(hits))
    focus = hits.loc[hits["hit_id"].isin(focus_hit_ids), XYZ].to_numpy(dtype=float) \
        if "hit_id" in hits else np.empty((0, 3))

    tail = (1 - quantile) / 2
    region = {}
    for axis, name in enumerate("xyz"):
        order = np.argsort(xyz[:, axis])
        cumulative = np.cumsum(weights[order]) / weights.sum()
        lo = xyz[order, axis][min(np.searchsorted(cumulative, tail), len(order) - 1)]
        hi = xyz[order, axis][min(np.searchsorted(cumulative, 1 - tail), len(order) - 1)]
        if len(focus):
            lo, hi = min(lo, focus[:, axis].min()), max(hi, focus[:, axis].max())
        span = max(hi - lo, 1.0)
        region[name] = (lo - pad * span, hi + pad * span)
    return region


def apply_zoom(hits, zoom_region):
    """(hits inside the box, box as (2, 3) array), or (hits, None)."""
    if zoom_region is None:
        return hits, None

    # Example structure: {"x": (xmin, xmax), "y": (...), "z": (...)}
    zoom_bounds = np.asarray(
        [zoom_region[axis] for axis in ("x", "y", "z")],
        dtype=float,
    ).T

    if (
        zoom_bounds.shape != (2, 3)
        or not np.isfinite(zoom_bounds).all()
        or np.any(zoom_bounds[1] <= zoom_bounds[0])
    ):
        raise ValueError(
            "zoom_region must contain finite, increasing "
            "(min, max) pairs for x, y and z."
        )

    positions = hits[XYZ].to_numpy()
    inside = (
        (positions >= zoom_bounds[0])
        & (positions <= zoom_bounds[1])
    ).all(axis=1)
    return hits.loc[inside], zoom_bounds


def hit_bounds(hits):
    xyz = hits[XYZ].to_numpy(dtype=float)
    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    return np.stack([xyz.min(axis=0), xyz.max(axis=0)]) if len(xyz) else None


def axis_bounds(view, zoom_bounds, detector_bounds, event_bounds):
    """(center, span) of the axes, or None."""
    if zoom_bounds is not None:
        # Explicit zoom overrides automatic event/detector bounds.
        return zoom_bounds.mean(axis=0), zoom_bounds[1] - zoom_bounds[0]

    if view == "detector":
        bounds = detector_bounds.copy()
        if event_bounds is not None:
            bounds[0] = np.minimum(bounds[0], event_bounds[0])
            bounds[1] = np.maximum(bounds[1], event_bounds[1])
    else:
        bounds = event_bounds if event_bounds is not None else detector_bounds
    if bounds is None:
        return None

    span = bounds[1] - bounds[0]
    span = np.maximum(span, max(span.max() * 0.02, 1e-6)) * 1.10
    return bounds.mean(axis=0), span


def view_title(title, n_shown, n_total, zoomed):
    return (f"{title}: {n_shown} / {n_total} hits in zoom region"
            if zoomed else f"{title}: {n_total} hits")


def marker_size(energy):
    return np.clip(8 + 12 * np.log10(1 + np.maximum(energy, 0)), 8, 75)


def particle_label(pid):
    try:
        name = Particle.from_pdgid(int(pid)).latex_name
    except (ValueError, LookupError, AttributeError):
        return f"PID {int(pid)}"
    return name if name.startswith("$") else f"${name}$"


def particle_name(pid):
    try:
        return Particle.from_pdgid(int(pid)).name
    except (ValueError, LookupError, AttributeError):
        return f"PID {int(pid)}"


def object_colour(object_id):
    return ("lightgray" if object_id < 0
            else mpl.colors.to_hex(plt.get_cmap("tab20")(int(object_id) % 20)))


# =============================================================================
# Detector drawing
# =============================================================================
DETECTOR_COLORS = {
    "PbWO4": "royalblue",
    "Brass": "darkgoldenrod",
    "Scintillator": "gray",
}


def detector_geometry(
    *,
    barrel_inner_radius_cm,
    barrel_half_length_cm,
    endcap_front_z_cm,
    endcap_inner_radius_cm,
    endcap_outer_radius_cm,
    coordinate_units_per_cm,
):
    """Build the projective calorimeter wireframe from the DICE geometry.

    Plotting-backend independent. Returns a dict with
      lines:    list of (material, (N, 3) polyline) in plot coordinates
      surfaces: list of (material, (K, 4, 3) quad faces) in plot coordinates
      bounds:   (2, 3) array of [min, max] xyz
    """
    dimensions = np.asarray([
        barrel_inner_radius_cm,
        barrel_half_length_cm,
        endcap_front_z_cm,
        endcap_inner_radius_cm,
        endcap_outer_radius_cm,
        coordinate_units_per_cm,
    ], dtype=float)

    if not np.all(np.isfinite(dimensions)):
        raise ValueError("All geometry dimensions and the scale must be supplied.")
    if (
        np.any(dimensions[[0, 1, 2, 4, 5]] <= 0)
        or endcap_inner_radius_cm < 0
        or endcap_outer_radius_cm <= endcap_inner_radius_cm
    ):
        raise ValueError("Invalid detector dimensions.")

    scale = coordinate_units_per_cm

    # Reconstructed production layer sequence, with thicknesses in cm.
    barrel_layers = (
        [("PbWO4", 30)]
        + [("Brass", 6), ("Scintillator", 6)] * 3
        + [("Brass", 6)]
    )
    endcap_layers = (
        [("PbWO4", 30)]
        + [("Brass", 15), ("Scintillator", 10)] * 3
        + [("Brass", 15)]
    )

    phi = np.linspace(0, 2 * np.pi, 129)
    spokes = np.linspace(0, 2 * np.pi, 8, endpoint=False)

    lines = []
    surfaces = []

    def ring_points(radius, z):
        return scale * np.column_stack([
            radius * np.cos(phi),
            radius * np.sin(phi),
            np.full_like(phi, z),
        ])

    def ring(radius, z, material):
        lines.append((material, ring_points(radius, z)))

    def connect_rings(r1, z1, r2, z2, material):
        points1 = ring_points(r1, z1)
        points2 = ring_points(r2, z2)

        # Each adjacent pair of angles defines one surface panel.
        surfaces.append((material, np.stack([
            points1[:-1],
            points1[1:],
            points2[1:],
            points2[:-1],
        ], axis=1)))

        for angle in spokes:
            radii = np.array([r1, r2])
            lines.append((material, scale * np.column_stack([
                radii * np.cos(angle),
                radii * np.sin(angle),
                [z1, z2],
            ])))

    # Barrel: each layer has a fixed half-length calculated at its
    # inner radius. Successive layers therefore become longer.
    radius = barrel_inner_radius_cm
    barrel_z_max = 0.0

    for material, thickness in barrel_layers:
        r_in = radius
        r_out = r_in + thickness
        half_length = (
            barrel_half_length_cm * r_in / barrel_inner_radius_cm
        )

        for r in (r_in, r_out):
            ring(r, -half_length, material)
            ring(r, +half_length, material)
            connect_rings(r, -half_length, r, +half_length, material)

        for z in (-half_length, +half_length):
            connect_rings(r_in, z, r_out, z, material)

        radius = r_out
        barrel_z_max = max(barrel_z_max, half_length)

    # Endcaps: both inner and outer radii increase with |z|.
    def endcap_radii(abs_z):
        factor = abs_z / endcap_front_z_cm
        return (
            endcap_inner_radius_cm * factor,
            endcap_outer_radius_cm * factor,
        )

    abs_z = endcap_front_z_cm

    for material, thickness in endcap_layers:
        z_front = abs_z
        z_back = z_front + thickness
        ri_front, ro_front = endcap_radii(z_front)
        ri_back, ro_back = endcap_radii(z_back)

        for sign in (-1, 1):
            for z, r_in, r_out in (
                (sign * z_front, ri_front, ro_front),
                (sign * z_back, ri_back, ro_back),
            ):
                ring(r_in, z, material)
                ring(r_out, z, material)
                connect_rings(r_in, z, r_out, z, material)

            # Sloping inner and outer surfaces.
            connect_rings(
                ri_front, sign * z_front,
                ri_back, sign * z_back, material,
            )
            connect_rings(
                ro_front, sign * z_front,
                ro_back, sign * z_back, material,
            )

        abs_z = z_back

    radius_max = max(radius, endcap_radii(abs_z)[1]) * scale
    z_max = max(barrel_z_max, abs_z) * scale

    return {
        "lines": lines,
        "surfaces": surfaces,
        "bounds": np.array([
            [-radius_max, -radius_max, -z_max],
            [+radius_max, +radius_max, +z_max],
        ]),
    }


def draw_detector_3d(
    ax,
    *,
    fill_layers=False,
    fill_alpha=0.04,
    wire_alpha=0.25,
    **geometry,
):
    """Draw the projective calorimeter using the supplied DICE geometry."""
    geom = detector_geometry(**geometry)

    # One line artist per material, NaN rows between the pieces: drawing
    # several hundred separate artists made every video frame ~3x slower.
    separator = np.full((1, 3), np.nan)
    for material, color in DETECTOR_COLORS.items():
        polylines = [np.vstack([p, separator]) for m, p in geom["lines"] if m == material]
        if not polylines:
            continue
        points = np.concatenate(polylines)
        ax.plot(
            points[:, 0], points[:, 1], points[:, 2],
            color=color,
            alpha=wire_alpha,
            lw=0.6,
            label="_nolegend_",
            zorder=1,  # under the hits (see computed_zorder in plot_event_hits_3d)
        )

    if fill_layers:
        for material, faces in geom["surfaces"]:
            ax.add_collection3d(
                Poly3DCollection(
                    faces,
                    facecolors=DETECTOR_COLORS[material],
                    edgecolors="none",
                    linewidths=0,
                    alpha=fill_alpha,
                    label="_nolegend_",
                    zorder=0,
                )
            )

    legend = ax.legend(
        handles=[
            Line2D([0], [0], color=color, lw=1.5, label=material)
            for material, color in DETECTOR_COLORS.items()
        ],
        title="Detector materials",
        loc="upper left",
        fontsize=8,
    )
    ax.add_artist(legend)

    return geom["bounds"]


def clip_polyline_to_box(points, bounds):
    """Clip an (N, 3) polyline to an XYZ box.

    Returns the surviving pieces as one (M, 3) array with NaN rows
    separating disconnected segments (empty if nothing is inside).
    """
    lower, upper = bounds
    points = np.asarray(points, dtype=float)
    pieces = []

    for start, end in zip(points[:-1], points[1:]):
        if not np.isfinite([start, end]).all():
            continue

        direction = end - start
        t_enter, t_exit = 0.0, 1.0

        for axis in range(3):
            if direction[axis] == 0:
                if not lower[axis] <= start[axis] <= upper[axis]:
                    t_enter, t_exit = 1.0, 0.0
                    break
            else:
                t1 = (lower[axis] - start[axis]) / direction[axis]
                t2 = (upper[axis] - start[axis]) / direction[axis]
                t_enter = max(t_enter, min(t1, t2))
                t_exit = min(t_exit, max(t1, t2))

            if t_enter > t_exit:
                break

        if t_enter <= t_exit:
            pieces.extend([
                start + t_enter * direction,
                start + t_exit * direction,
                [np.nan, np.nan, np.nan],
            ])

    return np.asarray(pieces).reshape(-1, 3)


def clip_detector_lines_to_box(ax, bounds):
    """Clip existing detector polylines to a physical XYZ box."""
    for artist in list(ax.lines):
        points = np.column_stack(artist.get_data_3d())
        clipped = clip_polyline_to_box(points, bounds)
        artist.set_data_3d(clipped[:, 0], clipped[:, 1], clipped[:, 2])


# =============================================================================
# Matplotlib renderer (static view and rotation frames)
# =============================================================================
PID_MARKERS = {
    22: "*",                  # photon
    11: "^",  -11: "^",       # electron / positron
    13: "v",  -13: "v",       # muon / antimuon
    211: "o", -211: "o",      # charged pion
    321: "D", -321: "D",      # charged kaon
    2212: "s", -2212: "s",    # proton / antiproton
    2112: "p", -2112: "p",    # neutron / antineutron
}


def plot_event_hits_3d(
    hits,
    *,
    title,
    colour_by=None,
    zmax=None,
    colour_label=None,
    ring_hit_ids=None,
    show_pid=False,
    detector=None,
    view="event",
    coordinate_unit=None,
    vertical_axis="z",
    elev=60, azim=100,
    zoom_region=None,
    show=False,
):
    """One event in 3D.

    colour_by: None colours hits by true object (the original view);
        a column of `hits` (e.g. "z_post_gn1") colours them on a symmetric
        diverging scale [-zmax, zmax] with a colour bar.
    ring_hit_ids: {"+": hit_id, "-": hit_id} (see extreme_hits); each hit
        gets a black ring in the shape given by RINGS.
    """
    if view not in {"event", "detector"}:
        raise ValueError("view must be 'event' or 'detector'")
    if view == "detector" and detector is None:
        raise ValueError("view='detector' requires a detector configuration")
    if colour_by is not None and not zmax:
        raise ValueError("colour_by needs a positive zmax")

    total_hits = len(hits)
    hits, zoom_bounds = apply_zoom(hits, zoom_region)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    # Draw in a fixed order (detector, hits, rings) instead of by depth;
    # otherwise the rings disappear behind the hits around them.
    ax.computed_zorder = False

    # Capture hit-only limits before drawing the detector.
    # This keeps the event view centred on the shower.
    event_bounds = hit_bounds(hits)
    detector_bounds = (
        draw_detector_3d(ax, **detector)
        if detector is not None else None
    )
    if zoom_bounds is not None and detector is not None:
        clip_detector_lines_to_box(ax, zoom_bounds)

    shape_handles = {}
    if colour_by is None:
        for object_id, object_hits in hits.groupby("t_idx"):
            color = object_colour(object_id)
            for pid, group in object_hits.groupby("t_pid"):
                marker = "x" if object_id < 0 else PID_MARKERS.get(int(pid), "X")
                label = (
                    "unassigned" if object_id < 0
                    else f"object {int(object_id)}: {particle_label(pid)}"
                )
                ax.scatter(
                    group["recHitX"], group["recHitY"], group["recHitZ"],
                    s=marker_size(group["recHitEnergy"]),
                    marker=marker, color=color, alpha=0.75,
                    label=label, depthshade=False, zorder=2,
                )
    else:
        norm = mpl.colors.Normalize(vmin=-zmax, vmax=zmax)
        cmap = plt.get_cmap(GRADIENT_CMAP)
        noise = hits["t_idx"].lt(0).rename("noise")
        for (is_noise, pid), group in hits.groupby([noise, "t_pid"]):
            marker = "x" if is_noise else PID_MARKERS.get(int(pid), "X")
            label = "unassigned" if is_noise else particle_label(pid)
            ax.scatter(
                group["recHitX"], group["recHitY"], group["recHitZ"],
                c=group[colour_by], cmap=cmap, norm=norm,
                s=marker_size(group["recHitEnergy"]),
                marker=marker, alpha=0.85,
                label="_nolegend_", depthshade=False, zorder=2,
            )
            shape_handles.setdefault(label, Line2D(
                [], [], marker=marker, linestyle="none", color="dimgray",
                markersize=7, label=label,
            ))

    if show_pid:
        for hit in hits.itertuples(index=False):
            ax.text(hit.recHitX, hit.recHitY, hit.recHitZ,
                    str(int(hit.t_pid)), fontsize=7)

    ring_handles = {}
    for sign, hit_id in (ring_hit_ids or {}).items():
        ringed = hits[hits["hit_id"] == hit_id] if "hit_id" in hits else hits.iloc[:0]
        if ringed.empty:  # outside the zoom box
            continue
        style = RINGS[sign]
        ax.scatter(
            ringed["recHitX"], ringed["recHitY"], ringed["recHitZ"],
            s=3 * marker_size(ringed["recHitEnergy"]) + 90,
            marker=style["mpl_marker"], facecolors="none", edgecolors="black",
            linewidths=1.4, label="_nolegend_", depthshade=False, zorder=3,
        )
        ring_handles[style["tex"]] = Line2D(
            [], [], marker=style["mpl_marker"], linestyle="none", markersize=9,
            markerfacecolor="none", markeredgecolor="black")

    unit_label = f" [{coordinate_unit}]" if coordinate_unit else ""
    ax.set(
        xlabel=f"recHitX{unit_label}",
        ylabel=f"recHitY{unit_label}",
        zlabel=f"recHitZ{unit_label}",
        title=view_title(title, len(hits), total_hits, zoom_bounds is not None),
    )
    ax.title.set_fontsize(10)
    ax.view_init(elev=elev, azim=azim, vertical_axis=vertical_axis)

    limits = axis_bounds(view, zoom_bounds, detector_bounds, event_bounds)
    if limits is not None:
        center, span = limits
        ax.set_xlim(center[0] - span[0] / 2, center[0] + span[0] / 2)
        ax.set_ylim(center[1] - span[1] / 2, center[1] + span[1] / 2)
        ax.set_zlim(center[2] - span[2] / 2, center[2] + span[2] / 2)
        ax.set_box_aspect(span)

    if not hits.empty:
        # Show representative energies from this event in a size legend.
        energies = hits["recHitEnergy"].clip(lower=0).to_numpy()
        examples = np.unique(np.quantile(energies, [0, 0.5, 1]))
        size_handles = [
            Line2D([], [], marker="o", linestyle="none",
                   markerfacecolor="white", markeredgecolor="black",
                   markersize=np.sqrt(marker_size(energy)))
            for energy in examples
        ]

        if colour_by is None:
            # Keep one entry for repeated labels, especially "unassigned".
            handles, labels = ax.get_legend_handles_labels()
            entries = dict(zip(labels, handles))
            entries.pop("_nolegend_", None)
            legend_title = "True objects"
        else:
            entries = {**shape_handles, **ring_handles}
            legend_title = "True PID (marker)"

        object_legend = fig.legend(
            entries.values(),
            entries.keys(),
            title=legend_title,
            ncol=2,
            loc="upper right",
            bbox_to_anchor=(0.99, 0.98),
            fontsize=8,
            columnspacing=0.8,
            handletextpad=0.3,
        )

        fig.canvas.draw()
        object_bottom = object_legend.get_window_extent(
            fig.canvas.get_renderer()
        ).transformed(fig.transFigure.inverted()).y0

        fig.legend(
            handles=size_handles,
            labels=[f"{energy:.3g}" for energy in examples],
            title="recHitEnergy\n(marker size)",
            loc="upper right",
            bbox_to_anchor=(0.99, object_bottom - 0.02),
            fontsize=8,
        )

    bottom = 0.05
    if colour_by is not None:
        bottom = 0.15
        cax = fig.add_axes([0.10, 0.07, 0.55, 0.02])
        colour_bar = fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=cax, orientation="horizontal", extend="both",
        )
        colour_bar.set_label(colour_label or colour_by, fontsize=9)
        colour_bar.ax.tick_params(labelsize=8)

    ax.tick_params(axis="both", which="major", labelsize=8)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.label.set_fontsize(10)

    fig.subplots_adjust(left=0.02, right=0.77, bottom=bottom, top=0.95)
    if show:
        plt.show()

    return fig, ax


def animate_rotation(
    fig, ax,
    *,
    elev=10,
    vertical_axis="y",
    start_azim=0,
    n_frames=60,
    fps=20,
):
    """Live rotation for interactive use (e.g. a notebook). For saving
    to disk use run_animation, which renders each frame only once."""
    angles = start_azim + np.linspace(
        0, 360, n_frames, endpoint=False
    )

    def update(azim):
        ax.view_init(
            elev=elev,
            azim=azim,
            vertical_axis=vertical_axis,
        )
        return ()

    return FuncAnimation(
        fig,
        update,
        frames=angles,
        interval=1000 / fps,
        blit=False,
        repeat=True,
    )


def find_ffmpeg():
    """Return an ffmpeg executable, or None.

    Prefers the static build shipped by `pip install imageio-ffmpeg`
    (always has libx264, works without root), then whatever matplotlib
    is configured to use / finds on PATH.
    """
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        pass
    return shutil.which(mpl.rcParams["animation.ffmpeg_path"])


def read_animation(path):
    """Frames (RGB) and frames per second of an animated webp or gif."""
    with Image.open(path) as image:
        frames, durations = [], []
        for frame in ImageSequence.Iterator(image):
            frames.append(frame.convert("RGB"))
            durations.append(frame.info.get("duration") or 100)
    return frames, 1000 / float(np.median(durations))


def render_rotation_frames(
    fig, ax,
    *,
    elev=10,
    vertical_axis="y",
    start_azim=0,
    n_frames=30,
    dpi=100,
):
    """Render one full turn of the camera and return the frames as RGB images.

    Rendering is the slow part, so it is done once here and the frames are
    then encoded to as many formats as needed by save_frames.
    """
    angles = start_azim + np.linspace(0, 360, n_frames, endpoint=False)
    frames = []

    # A "tight" bbox from the style sheet would change the frame size
    # from one camera angle to the next, which breaks video encoding.
    with plt.rc_context({"savefig.bbox": None}):
        for azim in angles:
            ax.view_init(elev=elev, azim=azim, vertical_axis=vertical_axis)
            buffer = io.BytesIO()
            fig.savefig(buffer, format="png", dpi=dpi, facecolor="white")
            frames.append(Image.open(buffer).convert("RGB"))

    return frames


def _render_frame_chunk(job):
    """Worker: draw the event once, render its share of camera angles as PNG bytes."""
    hits, plot_kwargs, angles, elev, vertical_axis, dpi = job
    fig, ax = plot_event_hits_3d(hits, show=False, **plot_kwargs)
    pngs = []
    with plt.rc_context({"savefig.bbox": None}):
        for azim in angles:
            ax.view_init(elev=elev, azim=azim, vertical_axis=vertical_axis)
            buffer = io.BytesIO()
            fig.savefig(buffer, format="png", dpi=dpi, facecolor="white")
            pngs.append(buffer.getvalue())
    plt.close(fig)
    return pngs


def render_rotation_frames_parallel(
    hits, plot_kwargs,
    *,
    elev=10,
    vertical_axis="y",
    start_azim=0,
    n_frames=30,
    dpi=100,
    workers=4,
):
    """Same frames as render_rotation_frames, rendered by `workers` processes.

    Each process draws the figure itself and renders a contiguous block of
    angles; the blocks are joined in order.
    """
    import multiprocessing

    angles = start_azim + np.linspace(0, 360, n_frames, endpoint=False)
    blocks = [block for block in np.array_split(angles, workers) if len(block)]
    jobs = [(hits, plot_kwargs, block, elev, vertical_axis, dpi) for block in blocks]
    # fork: the workers inherit the style set by setMplParam and the fonts.
    with multiprocessing.get_context("fork").Pool(len(jobs)) as pool:
        results = pool.map(_render_frame_chunk, jobs)
    return [Image.open(io.BytesIO(png)).convert("RGB") for block in results for png in block]


def part_path(path: Path) -> Path:
    """Hidden temporary name next to `path`, same extension (the writers read it).

    Outputs are written there and renamed when complete, so an interrupted run
    never leaves a truncated file that the next run would skip as done.
    """
    return path.with_name(f".{path.stem}.part{path.suffix}")


def save_frames(
    frames,
    save_path,
    *,
    formats=("mp4",),
    fps=10,
    crf=20,
    webp_quality=80,
):
    """Encode pre-rendered frames.

    formats:
      "mp4"  H.264 video. Smallest file by far; plays in any browser,
             VS Code, PowerPoint/Keynote, Mattermost, Indico. Needs ffmpeg
             (`pip install imageio-ffmpeg` if the cluster has none).
      "webp" Animated image. Shows inline wherever an image does
             (Markdown, Jupyter, chat, GitLab); no ffmpeg needed.
      "gif"  Same idea, maximum compatibility but larger and 256 colours.
    crf: H.264 quality (lower = better/larger; 18-23 is typical).
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    frame_ms = round(1000 / fps)
    saved = []

    for fmt in formats:
        path = save_path.with_suffix(f".{fmt}")
        part = part_path(path)

        if fmt == "mp4":
            ffmpeg = find_ffmpeg()
            if ffmpeg is None:
                print("ffmpeg not found (pip install imageio-ffmpeg) -> skipping mp4")
                continue

            width, height = frames[0].size
            command = [
                ffmpeg, "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
                # H.264 needs even frame dimensions.
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2:color=white",
                "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
                # yuv420p + faststart: plays in browsers, QuickTime, slides.
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                str(part),
            ]
            result = subprocess.run(
                command,
                input=b"".join(frame.tobytes() for frame in frames),
                capture_output=True,
            )
            if result.returncode != 0:
                part.unlink(missing_ok=True)
                print(
                    f"ffmpeg failed -> skipping mp4 "
                    f"(try `pip install imageio-ffmpeg`):\n"
                    f"{result.stderr.decode(errors='replace')}"
                )
                continue

        elif fmt in ("webp", "gif"):
            # method 4: same size as the slowest setting (6), ~10x faster.
            images, options = frames, {"quality": webp_quality, "method": 4}
            if fmt == "gif":
                # Pillow's default GIF palette drops small coloured areas
                # (legend lines turn red); octree keeps them.
                images = [
                    frame.quantize(
                        256,
                        method=Image.Quantize.FASTOCTREE,
                        dither=Image.Dither.NONE,
                    )
                    for frame in frames
                ]
                options = {"optimize": True}

            images[0].save(
                part,
                save_all=True,
                append_images=images[1:],
                duration=frame_ms,
                loop=0,
                **options,
            )

        else:
            raise ValueError(f"Unknown format {fmt!r}; use 'mp4', 'webp' or 'gif'.")

        os.replace(part, path)
        saved.append(path)
        print(f"Saved animation: {path} ({path.stat().st_size / 1e6:.1f} MB)")

    return saved


def run_animation(
    hits,
    *,
    save_path,
    formats=("mp4",),
    dpi=100,
    crf=20,
    webp_quality=80,
    animation_kwargs=None,
    workers=1,
    **plot_kwargs,
):
    """Render a rotating 3D event view and save it.

    save_path: output path without (or with any) suffix; the suffix is
        replaced per format, e.g. ".../zoom__post_gn1" -> zoom__post_gn1.mp4
    formats: any of "mp4", "webp", "gif" (see save_frames). The frames are
        rendered once, so extra formats only cost encoding time.
    animation_kwargs: elev, vertical_axis, start_azim, n_frames, fps.
    workers: processes rendering frames in parallel (1 = in this process).
    plot_kwargs: passed to plot_event_hits_3d.
    """
    import time

    # Keep the plot's camera settings unless explicitly overridden.
    rotation_kwargs = {
        "elev": plot_kwargs.get("elev", 60),
        "vertical_axis": plot_kwargs.get("vertical_axis", "z"),
        "start_azim": plot_kwargs.get("azim", 100),
    }
    rotation_kwargs.update(animation_kwargs or {})
    fps = rotation_kwargs.pop("fps", 10)

    start = time.time()
    if workers > 1:
        frames = render_rotation_frames_parallel(
            hits, plot_kwargs, dpi=dpi, workers=workers, **rotation_kwargs)
    else:
        fig, ax = plot_event_hits_3d(hits, show=False, **plot_kwargs)
        frames = render_rotation_frames(fig, ax, dpi=dpi, **rotation_kwargs)
        plt.close(fig)
    print(f"  rendered {len(frames)} frames in {time.time() - start:.0f} s "
          f"({workers} process{'es' if workers > 1 else ''})")

    return save_frames(
        frames, save_path,
        formats=formats,
        fps=fps,
        crf=crf,
        webp_quality=webp_quality,
    )


# =============================================================================
# Plotly renderer (interactive page)
# =============================================================================
# Plotly's 3D scatter only supports this small set of marker symbols.
PLOTLY_PID_SYMBOLS = {
    22: "diamond",                          # photon
    11: "cross", -11: "cross",              # electron / positron
    13: "square-open", -13: "square-open",  # muon / antimuon
    211: "circle", -211: "circle",          # charged pion
    321: "diamond-open", -321: "diamond-open",  # charged kaon
    2212: "square", -2212: "square",        # proton / antiproton
    2112: "circle-open", -2112: "circle-open",  # neutron / antineutron
}


# Turntable rotation for the plotly page. The data is stored once and
# only the camera moves, so the file stays as small as a static view.
# "{plot_id}" is filled in by plotly; __SPEED__ / __AUTOSTART__ by us.
PLOTLY_ROTATION_JS = """
var gd = document.getElementById('{plot_id}');
var speed = __SPEED__ * Math.PI / 180;  // radians per second
var running = false, last = null;

var button = document.createElement('button');
button.style.cssText =
    'position:absolute;left:10px;top:10px;z-index:10;padding:4px 10px;' +
    'font:13px sans-serif;cursor:pointer;';
gd.parentNode.style.position = 'relative';
gd.parentNode.insertBefore(button, gd);

function setRunning(on) {
    running = on;
    last = null;
    button.textContent = on ? 'Pause rotation' : 'Rotate';
}
button.onclick = function () { setRunning(!running); };

// Hand control to the user as soon as they drag the scene.
gd.addEventListener('pointerdown', function (event) {
    if (event.target.tagName === 'CANVAS') { setRunning(false); }
}, true);

function step(now) {
    if (running) {
        if (last !== null) {
            var camera = gd.layout.scene.camera;
            var e = camera.eye, u = camera.up;
            var norm = Math.hypot(u.x, u.y, u.z);
            var ux = u.x / norm, uy = u.y / norm, uz = u.z / norm;
            var angle = speed * (now - last) / 1000;
            var c = Math.cos(angle), s = Math.sin(angle);
            var dot = (ux * e.x + uy * e.y + uz * e.z) * (1 - c);
            // Rodrigues rotation of the eye about the vertical (up) axis.
            Plotly.relayout(gd, {'scene.camera.eye': {
                x: e.x * c + (uy * e.z - uz * e.y) * s + ux * dot,
                y: e.y * c + (uz * e.x - ux * e.z) * s + uy * dot,
                z: e.z * c + (ux * e.y - uy * e.x) * s + uz * dot
            }});
        }
        last = now;
    }
    requestAnimationFrame(step);
}
setRunning(__AUTOSTART__);
requestAnimationFrame(step);
"""


def plotly_colourscale(name, n=11):
    cmap = plt.get_cmap(name)
    return [[i / (n - 1), mpl.colors.to_hex(cmap(i / (n - 1)))] for i in range(n)]


def save_event_hits_3d_html(
    hits,
    *,
    save_path,
    title,
    stages,
    zmax,
    log_scale,
    ring_hit_ids=None,
    detector=None,
    view="event",
    coordinate_unit=None,
    vertical_axis="z",
    elev=60, azim=100,
    zoom_region=None,
    wire_opacity=0.35,
    hit_marker_scale=1.5,
    include_plotlyjs=True,
    auto_rotate=True,
    rotation_deg_per_s=10,
):
    """Save an interactive (rotate/zoom/pan/hover) plotly view of one event.

    stages: list of (stage key, label); `hits` must hold z_<key> and g_<key>.
    ring_hit_ids: {stage key: {"+": hit_id, "-": hit_id}} (see extreme_hits).
    A menu colours the hits by any stage or by true object. Legend entries
    are true objects and detector materials; click them to hide/show.
    Hovering a hit shows its gradient at every stage.
    include_plotlyjs: True embeds plotly.js (~4 MB, works offline);
        "cdn" keeps the file small but needs internet to open.
    auto_rotate: start a turntable rotation when the page opens. A button
        toggles it; dragging the scene stops it.
    """
    import plotly.graph_objects as go

    if view not in {"event", "detector"}:
        raise ValueError("view must be 'event' or 'detector'")
    if view == "detector" and detector is None:
        raise ValueError("view='detector' requires a detector configuration")

    total_hits = len(hits)
    hits, zoom_bounds = apply_zoom(hits, zoom_region)
    event_bounds = hit_bounds(hits)

    traces = []
    detector_bounds = None

    if detector is not None:
        # Drop matplotlib-only styling options if present.
        geom = detector_geometry(**{
            key: value for key, value in detector.items()
            if key not in {"fill_layers", "fill_alpha", "wire_alpha"}
        })
        detector_bounds = geom["bounds"]
        separator = np.full((1, 3), np.nan)

        for material, color in DETECTOR_COLORS.items():
            polylines = [p for m, p in geom["lines"] if m == material]
            if zoom_bounds is not None:
                polylines = [clip_polyline_to_box(p, zoom_bounds) for p in polylines]
            polylines = [np.vstack([p, separator]) for p in polylines if len(p)]
            if not polylines:
                continue

            # One trace per material; NaN rows break the line between pieces.
            points = np.concatenate(polylines)
            traces.append(go.Scatter3d(
                x=points[:, 0], y=points[:, 1], z=points[:, 2],
                mode="lines",
                line=dict(color=color, width=2),
                opacity=wire_opacity,
                name=material,
                legendgroup="detector",
                legendgrouptitle_text="Detector materials",
                hoverinfo="skip",
            ))

    def plotly_marker_size(energy):
        # Same area scale as the matplotlib plot; plotly sizes are diameters.
        return hit_marker_scale * np.sqrt(marker_size(energy))

    unit_label = f" [{coordinate_unit}]" if coordinate_unit else ""
    first_stage = stages[0][0]
    stage_hover = "".join(
        f"<br>g {label} = %{{customdata[{3 + i}]:.3e}}"
        for i, (_, label) in enumerate(stages)
    )

    # Per hit trace: its index, its object colour and its z values per stage.
    hit_traces, object_colours = [], []
    trace_z = {key: [] for key, _ in stages}
    shown_labels = set()
    for object_id, object_hits in hits.groupby("t_idx"):
        for pid, group in object_hits.groupby("t_pid"):
            symbol = "x" if object_id < 0 else PLOTLY_PID_SYMBOLS.get(int(pid), "x")
            label = (
                "unassigned" if object_id < 0
                else f"object {int(object_id)}: {particle_name(pid)}"
            )
            energy = group["recHitEnergy"].to_numpy(dtype=float)
            customdata = np.column_stack(
                [group["hit_id"], energy, group["t_pid"]]
                + [group[f"g_{key}"] for key, _ in stages]
            )

            hit_traces.append(len(traces))
            object_colours.append(object_colour(object_id))
            for key, _ in stages:
                trace_z[key].append(group[f"z_{key}"].round(4).tolist())
            traces.append(go.Scatter3d(
                x=group["recHitX"], y=group["recHitY"], z=group["recHitZ"],
                mode="markers",
                marker=dict(
                    size=plotly_marker_size(energy),
                    symbol=symbol,
                    color=trace_z[first_stage][-1],
                    coloraxis="coloraxis",
                    opacity=0.8,
                    line=dict(width=0),
                ),
                name=label,
                legendgroup=label,
                # "unassigned" spans several PIDs; show it once in the legend.
                showlegend=label not in shown_labels,
                customdata=customdata,
                hovertemplate=(
                    f"{label}<br>"
                    "hit %{customdata[0]:d}, recHitEnergy=%{customdata[1]:.3g}, "
                    "t_pid=%{customdata[2]:d}<br>"
                    f"x=%{{x:.1f}}, y=%{{y:.1f}}, z=%{{z:.1f}}{unit_label}"
                    f"{stage_hover}"
                    "<extra></extra>"
                ),
            ))
            shown_labels.add(label)

    # Two rings per stage: the hits with the largest +z and the most negative z
    # at that stage. Only the rings of the stage chosen in the menu are visible.
    ring_ids, ring_stage = [], []
    for i, (key, label) in enumerate(stages):
        for sign, hit_id in (ring_hit_ids or {}).get(key, {}).items():
            ring = hits[hits["hit_id"] == hit_id]
            if ring.empty:  # outside the zoom box
                continue
            style = RINGS[sign]
            ring_ids.append(len(traces))
            ring_stage.append(key)
            traces.append(go.Scatter3d(
                x=ring["recHitX"], y=ring["recHitY"], z=ring["recHitZ"],
                mode="markers",
                marker=dict(size=2.5 * plotly_marker_size(ring["recHitEnergy"]) + 4,
                            symbol=style["plotly_symbol"], color="black",
                            line=dict(width=3)),
                name=f"{style['label']} {label}",
                legendgroup="rings",
                visible=(i == 0),
                customdata=ring[["hit_id", f"g_{key}", f"z_{key}"]],
                hovertemplate=(f"{style['label']} {label}: hit %{{customdata[0]:d}}<br>"
                               "g = %{customdata[1]:.3e}, z = %{customdata[2]:.2f}"
                               "<extra></extra>"),
            ))

    def ring_visibility(shown_key):
        return [key == shown_key for key in ring_stage]

    # Menu: colour by each stage (and show that stage's rings), or by true object.
    scale_text = f"sign(g)·log10(1+|g|/{log_scale:g})"
    n_hit = len(hit_traces)
    buttons = [
        dict(
            label=label,
            method="update",
            args=[
                {"marker.color": trace_z[key] + ["black"] * len(ring_ids),
                 "marker.coloraxis": ["coloraxis"] * n_hit + [None] * len(ring_ids),
                 "visible": [True] * n_hit + ring_visibility(key)},
                {"coloraxis.showscale": True,
                 "coloraxis.colorbar.title.text": f"z {label}<br><sup>{scale_text}</sup>"},
                hit_traces + ring_ids,
            ],
        )
        for key, label in stages
    ]
    buttons.append(dict(
        label="true object",
        method="update",
        args=[
            {"marker.color": object_colours + ["black"] * len(ring_ids),
             "marker.coloraxis": [None] * (n_hit + len(ring_ids)),
             "visible": [True] * n_hit + ring_visibility(None)},
            {"coloraxis.showscale": False},
            hit_traces + ring_ids,
        ],
    ))

    scene = dict(
        xaxis_title=f"recHitX{unit_label}",
        yaxis_title=f"recHitY{unit_label}",
        zaxis_title=f"recHitZ{unit_label}",
    )

    limits = axis_bounds(view, zoom_bounds, detector_bounds, event_bounds)
    if limits is not None:
        center, span = limits
        for axis, c, s in zip("xyz", center, span):
            scene[f"{axis}axis_range"] = [c - s / 2, c + s / 2]
        scene["aspectmode"] = "manual"
        scene["aspectratio"] = dict(zip("xyz", span / span.max()))

    # Translate matplotlib's (elev, azim, vertical_axis) into a plotly camera.
    vertical = "xyz".index(vertical_axis)
    e, a = np.radians([elev, azim])
    eye = 2.0 * np.roll(
        [np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)],
        vertical - 2,
    )
    scene["camera"] = dict(
        eye=dict(zip("xyz", eye)),
        up=dict(zip("xyz", np.eye(3)[vertical])),
    )

    fig = go.Figure(traces)
    fig.update_layout(
        title=view_title(title, len(hits), total_hits, zoom_bounds is not None),
        title_x=0.5,  # keeps the title clear of the rotation button
        scene=scene,
        template="plotly_white",
        legend=dict(itemsizing="constant", groupclick="toggleitem"),
        coloraxis=dict(
            colorscale=plotly_colourscale(GRADIENT_CMAP),
            cmin=-zmax, cmax=zmax,
            colorbar=dict(
                title=dict(text=f"z {stages[0][1]}<br><sup>{scale_text}</sup>",
                           side="top"),
                orientation="h", x=0.5, xanchor="center",
                y=0.0, yanchor="top", len=0.5, thickness=14,
            ),
        ),
        updatemenus=[dict(
            buttons=buttons, direction="down", showactive=True, active=0,
            x=0.99, xanchor="right", y=1.0, yanchor="bottom",
        )],
        height=850,
        margin=dict(l=0, r=0, t=60, b=90),
    )

    rotation_js = (
        PLOTLY_ROTATION_JS
        .replace("__SPEED__", repr(float(rotation_deg_per_s)))
        .replace("__AUTOSTART__", "true" if auto_rotate else "false")
    )

    html_path = Path(save_path).with_suffix(".html")
    html_path.parent.mkdir(parents=True, exist_ok=True)
    part = part_path(html_path)
    fig.write_html(
        part,
        include_plotlyjs=include_plotlyjs,
        post_script=rotation_js,
    )
    os.replace(part, html_path)
    print(
        f"Saved interactive view: {html_path} "
        f"({html_path.stat().st_size / 1e6:.1f} MB)"
    )
    return html_path


# =============================================================================
# Driver
# =============================================================================
def event_dir_name(row) -> str:
    return f"{Path(row.root_file).stem}_e{int(row.entry):04d}"


def stage_dir_name(stages, stage) -> str:
    """Stage folder, numbered in network order so it sorts that way."""
    return f"{stages.index(stage) + 1}_{stage}"


def link_event(run_dir, stage_dir, rank, event_dir):
    """stages/<stage_dir>/<rank>_<event> -> ../../events/<event> (relative link)."""
    link = run_dir / "stages" / stage_dir / f"{rank:02d}_{event_dir.name}"
    target = Path("..", "..", event_dir.parent.name, event_dir.name)
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if Path(os.readlink(link)) == target:
            return
        link.unlink()
    elif link.exists():
        raise FileExistsError(f"{link} exists and is not a link; move it away")
    os.symlink(target, link, target_is_directory=True)


def remove_stale_links(run_dir, wanted):
    """Remove links in stages/*/ that are not in `wanted` (links only, never files)."""
    for link in (run_dir / "stages").glob("*/*"):
        if link.is_symlink() and (link.parent.name, link.name) not in wanted:
            link.unlink()
            print(f"  removed old link stages/{link.parent.name}/{link.name}")


def view_settings(name, hits, focus_hit_ids):
    settings = dict(VIEWS[name], coordinate_unit=COORDINATE_UNIT)
    if settings.get("zoom_region") == "auto":
        settings["zoom_region"] = auto_zoom_region(hits, focus_hit_ids=focus_hit_ids)
    return settings


def pick_events_per_stage(selection, stages, n) -> pd.DataFrame:
    """Per stage, the n selection events with the largest |g| at that stage.

    z = sign(g) log10(1 + |g|/scale) grows with |g|, so ranking by |z| gives
    the same events in the same order.

    Only hits that are top-n at that stage (listed in top_in) compete, so a
    stage's picks are its own strongest events: in gradient_statistics each
    event enters a stage's top-n with its hit of largest |g| at that stage.
    Returns one row per pick: stage, rank (1 = largest), event_id, hit_id, g.
    """
    top_in = selection["top_in"].astype(str).str.split(";")
    picks = []
    for stage in stages:
        rows = selection[top_in.apply(lambda names: stage in names)]
        rows = rows.loc[rows[stage].abs().sort_values(ascending=False).index]
        rows = rows.drop_duplicates("event_id").head(n)
        for rank, row in enumerate(rows.itertuples(index=False), start=1):
            picks.append({"stage": stage, "rank": rank, "event_id": int(row.event_id),
                          "hit_id": int(row.hit_id), "g": float(getattr(row, stage))})
    return pd.DataFrame(picks, columns=["stage", "rank", "event_id", "hit_id", "g"])


def stages_to_animate(spec, picked_stages, stages, hits):
    """Stages that get a rotation video, from --animate."""
    if spec == ["none"]:
        return []
    if spec == ["all"]:
        return list(stages)
    if spec == ["selected"] and picked_stages:
        # The stages for which this event was picked (pick_events_per_stage).
        return [s for s in stages if s in picked_stages]
    if spec in (["peak"], ["selected"]):
        # One stage: where this event's largest |z| is, from its own gradients.
        # Also the fallback for events that were not picked for any stage.
        return [max(stages, key=lambda s: hits[f"z_{s}"].abs().max())]
    unknown = sorted(set(spec) - set(stages))
    if unknown:
        raise ValueError(f"Unknown stages for --animate: {unknown}; known: {stages}")
    return [s for s in stages if s in spec]


def render_event(event_id, *, row, gradients, converter, stages, log_scale, zmax,
                 selection, picked_ranks, args):
    """picked_ranks: {stage: rank} for the stages this event was picked for."""
    out_dir = args.run_dir / "events" / event_dir_name(row)
    event_rows = (selection[selection["event_id"] == event_id]
                  if selection is not None else pd.DataFrame())
    selected_hit_ids = [int(h) for h in event_rows.get("hit_id", [])]
    print(f"\nEvent {event_id}: {Path(row.root_file).name}, entry {int(row.entry)}"
          "; picked for " + (", ".join(f"{s} (rank {r})" for s, r in picked_ranks.items())
                             or "no stage"))

    if int(row.entry) < 0:
        print("  skipped: entry unknown (check L3 failed for this file)")
        return

    model_hits = load_djctd_event(row.djctd_file, int(row.local_idx))
    converted = load_converted_event(row.root_file, int(row.entry), converter)
    event_df, l5 = attach_truth(model_hits, converted)
    info = {
        "event_id": int(event_id),
        "root_file": row.root_file,
        "entry": int(row.entry),
        "djctd_file": row.djctd_file,
        "local_idx": int(row.local_idx),
        "n_hits": int(row.n_hits),
        "selected_hit_ids": selected_hit_ids,
        "picked_for_stages": {s: int(r) for s, r in picked_ranks.items()},
        "checks": {"L5": l5},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "event_info.json", "w", encoding="utf-8") as handle:
        json.dump(info, handle, indent=2)

    print(f"  L5: {l5['status']}: {l5['hits_found_in_root_entry']} / {l5['djctd_hits']} "
          f"hits found in the root entry ({l5['converter_hits']} converted), same energy "
          f"{l5.get('same_energy', float('nan')):.3f}; t_idx from {l5['t_idx_from']}, "
          f"t_pid from {l5['t_pid_from']}  {l5.get('reason', '')}")
    if l5["status"] != "passed":
        print("  not drawn")
        return
    if event_df.empty:
        print("  no hits, nothing to draw")
        return

    hits = annotate_hits(event_df, gradients[event_id], stages, log_scale)
    hits.to_csv(out_dir / "annotated_hits.csv", index=False)

    # The rings of each stage, from this event's own gradients: the hit with the
    # largest positive z and the hit with the most negative z.
    rings = {s: extreme_hits(hits, s) for s in stages}
    by_id = hits.set_index("hit_id")
    info["ringed_hits"] = {
        s: {RINGS[sign]["label"]: {"hit_id": h,
                                   "g": float(by_id.at[h, f"g_{s}"]),
                                   "z": float(by_id.at[h, f"z_{s}"])}
            for sign, h in ext.items()}
        for s, ext in rings.items()
    }
    with open(out_dir / "event_info.json", "w", encoding="utf-8") as handle:
        json.dump(info, handle, indent=2)
    print(f"  ringed hits (hit_id 0..{len(hits) - 1}):")
    for s, ext in rings.items():
        print(f"    {s:<18}" + "   ".join(
            f"{RINGS[sign]['label']}: hit {h} (z={by_id.at[h, f'z_{s}']:+.2f})"
            for sign, h in ext.items()))
    ringed_ids = sorted({h for ext in rings.values() for h in ext.values()})

    if zmax is None:  # no selection: scale from this event alone
        zmax = float(max(hits[f"z_{s}"].abs().max() for s in stages)) or 1.0
    labels = {s: STAGE_LABELS.get(s, s) for s in stages}
    title = f"Event {event_id} ({Path(row.root_file).stem}, entry {int(row.entry)})"
    scale_tex = latex_scientific(log_scale)

    # One full turn at --rotation-speed degrees per second, played at --fps.
    rotation = {"fps": args.fps,
                "n_frames": max(2, round(360 * args.fps / args.rotation_speed))}

    def done(path, formats):
        """True if every output exists and --overwrite was not given."""
        return not args.overwrite and all(
            Path(path).with_suffix(f".{fmt}").is_file() for fmt in formats)

    for view in args.views:
        # The page shows every stage, so its zoom box keeps every stage's rings inside.
        settings = view_settings(view, hits, ringed_ids)
        html_path = out_dir / f"interactive_{view}"
        if done(html_path, ["html"]):
            print(f"  exists, skipped: {html_path.name}.html")
        else:
            save_event_hits_3d_html(
                hits,
                save_path=html_path,
                title=title,
                stages=[(s, labels[s]) for s in stages],
                zmax=zmax,
                log_scale=log_scale,
                ring_hit_ids=rings,
                rotation_deg_per_s=args.rotation_speed,
                **settings,
            )
        for stage in stages_to_animate(args.animate, picked_ranks, stages, hits):
            # Named by stage position, so an event's videos sort in network order.
            video_path = out_dir / f"{stage_dir_name(stages, stage)}__{view}"
            wanted = [f for f in args.formats
                      if args.overwrite or not video_path.with_suffix(f".{f}").is_file()]
            if not wanted:
                print(f"  exists, skipped: {video_path.name}.{'/'.join(args.formats)}")
                continue
            # The same video in another format: re-encode its frames, no rendering.
            source = next((video_path.with_suffix(f".{f}") for f in ("webp", "gif")
                           if f not in wanted and video_path.with_suffix(f".{f}").is_file()),
                          None)
            if source is not None:
                frames, fps = read_animation(source)
                print(f"  {video_path.name}: {'/'.join(wanted)} from the existing {source.name}")
                save_frames(frames, video_path, formats=tuple(wanted), fps=fps)
                continue
            run_animation(
                hits,
                save_path=video_path,
                formats=tuple(wanted),
                animation_kwargs=rotation,
                workers=args.workers,
                title=f"{title} {labels[stage]}",
                colour_by=f"z_{stage}",
                zmax=zmax,
                colour_label=(rf"$z=\mathrm{{sign}}(g)\,\log_{{10}}"
                              rf"(1+|g|/{scale_tex})$ {labels[stage]}"),
                ring_hit_ids=rings[stage],
                # Same camera and zoom box as the other stages of this event.
                **settings,
            )

    # Link the event from every stage it was picked for, named by rank.
    for stage, rank in picked_ranks.items():
        link_event(args.run_dir, stage_dir_name(stages, stage), rank, out_dir)


def run(args):
    if "mp4" in args.formats and find_ffmpeg() is None:
        raise SystemExit(
            "mp4 needs ffmpeg, which was not found. Install it once (into ~/.local, "
            "which the container sees) with\n"
            "  apptainer exec /ceph/cjo/containers/plotting.sif pip install --user imageio-ffmpeg\n"
            "or leave mp4 out: --formats webp")
    index = read_event_index(args.run_dir)
    selection, selection_meta = find_selection(args.run_dir, args.selection)

    if selection_meta is not None:
        stage_files = {s["key"]: Path(s["file"]) for s in selection_meta["stages"]}
        log_scale = selection_meta["transform"]["scale"]
        zmax = max(selection_meta["colour_scale"]["abs_z_p999"].values())
    elif args.probe_dir is not None:
        stage_files = discover_stages(args.probe_dir)
        log_scale, zmax = args.log_scale, None
    else:
        raise ValueError("No selection in the run directory: pass --probe-dir.")
    if args.zmax is not None:
        zmax = args.zmax
    stages = list(stage_files)

    # Per stage, the events with the largest |g| at that stage.
    picks = (pick_events_per_stage(selection, stages, args.events_per_stage)
             if selection is not None else pd.DataFrame(columns=["stage", "event_id"]))
    picked_ranks = {e: dict(zip(g["stage"], g["rank"])) for e, g in picks.groupby("event_id")}

    if args.event_id:
        event_ids = list(dict.fromkeys(args.event_id))
    elif selection is not None:
        event_ids = list(dict.fromkeys(picks["event_id"].tolist()))
        if not event_ids:
            raise ValueError("No events picked: the selection has no top_in entries.")
    else:
        raise ValueError("No selection in the run directory: pass --event-id.")
    missing = [e for e in event_ids if e not in index.index]
    if missing:
        raise KeyError(f"event_ids not in events.csv: {missing}")

    if not args.event_id and not picks.empty:
        # Which event folder holds the top events of each stage.
        picks = picks.assign(
            z=signed_log10(picks["g"].to_numpy(dtype=float), scale=log_scale),
            folder=[event_dir_name(index.loc[e]) for e in picks["event_id"]])
        picks = picks[["stage", "rank", "event_id", "hit_id", "g", "z", "folder"]]
        (args.run_dir / "stages").mkdir(parents=True, exist_ok=True)
        picks.to_csv(args.run_dir / "stages" / "picked_by_stage.csv", index=False)
        remove_stale_links(args.run_dir, {
            (stage_dir_name(stages, r.stage), f"{r.rank:02d}_{r.folder}")
            for r in picks.itertuples()})
        print(f"Picked per stage ({args.events_per_stage} each, by |z| at that stage):")
        for stage in stages:
            group = picks[picks["stage"] == stage]
            print(f"  {stage:<18}" + ", ".join(
                f"{e} ({f})" for e, f in zip(group["event_id"], group["folder"])))
            if len(group) < args.events_per_stage:
                print(f"  WARNING: only {len(group)} events are top-n at {stage} in this "
                      "selection; rerun gradient_statistics with a larger --top-n and "
                      "--min-stages 1.")

    print(f"Events: {event_ids}\nStages: {stages}\n"
          f"Colour scale: |z| <= {zmax if zmax is not None else 'per event'}")
    gradients = read_event_gradients(stage_files, event_ids)
    converter = load_converter(args.converter)

    for event_id in event_ids:
        render_event(event_id, row=index.loc[event_id], gradients=gradients,
                     converter=converter, stages=stages, log_scale=log_scale,
                     zmax=zmax, selection=selection,
                     picked_ranks=picked_ranks.get(event_id, {}), args=args)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="3D event views coloured by per-hit probe gradients.",
    )
    parser.add_argument("--run-dir", required=True, type=Path,
                        help="run directory with events.csv and selections/")
    parser.add_argument("--selection", default=None,
                        help="selection name under selections/ (default: the only one)")
    parser.add_argument("--event-id", type=int, nargs="+", default=None,
                        help="event_ids to draw (default: events of the selection)")
    parser.add_argument("--events-per-stage", "--max-events", dest="events_per_stage",
                        type=int, default=10,
                        help="per stage, the events of the selection with the largest |z| "
                             "at that stage (default 10); at most the selection's --top-n")
    parser.add_argument("--views", nargs="+", default=list(VIEWS), choices=list(VIEWS))
    parser.add_argument("--animate", nargs="+", default=["all"],
                        help="stages with rotation videos, per event: all (default: "
                             "every stage), selected (only the stages the event was "
                             "picked for), peak (the stage of the event's largest |z|), "
                             "none, or stage names")
    parser.add_argument("--formats", nargs="+", default=["webp", "mp4"],
                        choices=["mp4", "webp", "gif"],
                        help="formats of the rotation videos (default webp mp4: webp for "
                             "VS Code, mp4 for Keynote / PowerPoint)")
    parser.add_argument("--rotation-speed", type=float, default=10.0,
                        help="degrees per second, in the videos and the HTML page (default 10)")
    parser.add_argument("--fps", type=int, default=10,
                        help="frames per second of the videos (default 10); one turn has "
                             "360 * fps / rotation-speed frames")
    parser.add_argument("--workers", type=int, default=4,
                        help="processes rendering video frames in parallel (default 4)")
    parser.add_argument("--overwrite", action="store_true",
                        help="redo outputs that already exist (default: skip them)")
    parser.add_argument("--zmax", type=float, default=None,
                        help="colour scale limit in z (default: from selection_meta.json)")
    parser.add_argument("--probe-dir", type=Path, default=None,
                        help="probe CSV directory, only needed without a selection")
    parser.add_argument("--log-scale", type=float, default=1e-8,
                        help="signed_log10 scale, only used without a selection")
    parser.add_argument("--converter", type=Path, default=DEFAULT_CONVERTER,
                        help="file defining TrainData_DICE_cellparent")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    # plot style settings
    setMplParam()
    font_dir = Path(mpl.get_data_path()) / "fonts" / "ttf"

    for font_path in font_dir.glob("*.ttf"):
        fm.fontManager.addfont(str(font_path))

    run(args)


if __name__ == "__main__":
    main()