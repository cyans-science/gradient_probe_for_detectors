"""
gradient_statistics.py

Purpose
-------
Distribution and selection of the per-hit probe gradients of one run.

  * Reads every stage CSV (event_id,hit_id,value) in --probe-dir. Stage names
    come from the file names, ordered from input to output head.
  * Checks that all stages list the same hits in the same order.
  * Plots the signed-log10 gradient distributions of all stages (overlay).
  * Selects hits: per stage, the hit with the largest |gradient| of each
    event, then the top-n events; kept if in the top-n of >= min_stages
    stages. Plots how the selected hits evolve across stages.
  * Writes selection.csv, the hand-off to gradient_hit_viewer.py. If an
    events.csv from build_event_index.py with a passed L1 check is found,
    each selected hit also gets its root file and entry.

Outputs (in --out-dir, the run directory)
-----------------------------------------
figures/gradient_histograms_overlay.{png,pdf}
selections/<name>/selection.csv
selections/<name>/selection_meta.json   criteria, stages, transform, colour scale
selections/<name>/top_hits_evolution.{png,pdf}

Gradients are used as written by fisher_val_loop, without per-batch
normalisation.

cyan.jo, 2026-10-21 (restructured from the earlier analysis script)
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Utils (PlotUtils, ExternalFunctions) is put on PYTHONPATH by run_gradient_statistics.sh.
try:
    from PlotUtils import setMplParam, getColour, getHistoParam, annotate_histogram, enable_histogram_counts
    from ExternalFunctions import add_text_to_ax, nice_string_output
except ModuleNotFoundError as error:
    raise SystemExit(
        f"{error}. PlotUtils and ExternalFunctions live in /work/cjo/Utils; run this "
        "script through its run_*.sh wrapper, which puts that folder on PYTHONPATH."
    ) from error

# Probe locations in forward order, and their plot labels.
STAGE_ORDER = [
    "input", "post_initial",
    "post_gn1", "post_gn2", "post_gn3", "post_gn4",
    "post_output_head",
]
STAGE_LABELS = {
    "input": "@Input",
    "post_initial": "@Initial Processing",
    "post_gn1": "@After GN1",
    "post_gn2": "@After GN2",
    "post_gn3": "@After GN3",
    "post_gn4": "@After GN4",
    "post_output_head": "@After Output Head",
}
EVENT_INDEX_COLUMNS = ["root_file", "entry", "djctd_file", "local_idx", "n_hits"]


# -----------------------------------------------------------------------------
# Reading and checking the probe tables
# -----------------------------------------------------------------------------
def discover_stages(probe_dir) -> dict:
    """Stage key -> CSV path for every CSV in probe_dir, in forward order.

    A file belongs to the longest known stage key contained in its name
    (so prefixed names like 'run_post_gn1.csv' work). Unknown names are kept
    under their own stem and placed after the known stages.
    """
    files = sorted(Path(probe_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files in {probe_dir}")

    keys_longest_first = sorted(STAGE_ORDER, key=len, reverse=True)
    stages = {}
    for path in files:
        key = next((k for k in keys_longest_first if k in path.stem), path.stem)
        if key in stages:
            raise ValueError(f"{path.name} and {stages[key].name} both match stage {key!r}")
        stages[key] = path

    def position(key):
        return (STAGE_ORDER.index(key) if key in STAGE_ORDER else len(STAGE_ORDER), key)

    unknown = [k for k in stages if k not in STAGE_ORDER]
    if unknown:
        print(f"WARNING: unknown stage names {unknown}, placed after the known stages.")
    return {key: stages[key] for key in sorted(stages, key=position)}


def read_probe_file(probe_file):
    """
    Read a Fisher/probe gradient CSV file.

    Expected format:
        event_id,hit_id,value
        0,0,0.0067
        0,1,-0.00058
        ...

    Returns
    -------
    pd.DataFrame
        Columns:
            event_id : int32
            hit_id   : int32
            value    : float32 (the gradients are float32 tensors, so this is exact)
    """
    probe_file = Path(probe_file)

    if not probe_file.exists():
        raise FileNotFoundError(f"Probe file not found: {probe_file}")

    df = pd.read_csv(
        probe_file,
        dtype={"event_id": "int32", "hit_id": "int32", "value": "float32"},
    )

    expected_columns = {"event_id", "hit_id", "value"}
    missing_columns = expected_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"{probe_file.name} is missing columns: {sorted(missing_columns)}"
        )

    return df[["event_id", "hit_id", "value"]]


def check_aligned(stage_frames: dict) -> None:
    """All stages must hold the same hits in the same row order.

    fisher_val_loop writes rows sorted by event_id with hit_id = 0..n-1 inside
    each event, identically for every stage. Later steps rely on this to
    compare stages row by row.
    """
    stages = list(stage_frames)
    reference = stage_frames[stages[0]]
    event_id = reference["event_id"].to_numpy()
    hit_id = reference["hit_id"].to_numpy()
    if event_id.size == 0:
        raise ValueError(f"{stages[0]}: no rows")
    if np.any(np.diff(event_id) < 0):
        raise ValueError(f"{stages[0]}: rows are not sorted by event_id")

    starts = np.flatnonzero(np.r_[True, event_id[1:] != event_id[:-1]])
    lengths = np.diff(np.r_[starts, event_id.size])
    expected_hit_id = np.arange(event_id.size) - np.repeat(starts, lengths)
    if not np.array_equal(hit_id, expected_hit_id):
        raise ValueError(f"{stages[0]}: hit_id is not 0..n-1 within each event")

    for stage in stages[1:]:
        frame = stage_frames[stage]
        if (
            len(frame) != len(reference)
            or not np.array_equal(frame["event_id"].to_numpy(), event_id)
            or not np.array_equal(frame["hit_id"].to_numpy(), hit_id)
        ):
            raise ValueError(f"{stage}: rows differ from {stages[0]}")


# -----------------------------------------------------------------------------
# Selection
# -----------------------------------------------------------------------------
def get_highest_values(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Top-n rows among the per-event maxima, ranked by |value| (sign kept)."""
    score = df["value"].abs()
    idx = score.groupby(df["event_id"]).idxmax()          # per-event max |value|
    return df.loc[score.loc[idx].nlargest(n).index]       # top-n events, signed rows


def track_top_hits(
    df_dict: dict, n: int = 20, min_stages: int = 1
) -> pd.DataFrame:
    """
    Follow the hits that are in the top-n (by |value|) at any stage across all stages.

    Parameters
    ----------
    df_dict : dict[str, pd.DataFrame]
        Stage name -> DataFrame (event_id, hit_id, value), in chronological order.
        All frames must be row-aligned (see check_aligned).
    n : int
        Number of top hits taken from each stage.
    min_stages : int
        Keep only hits that are in the top-n of at least this many stages.

    Returns
    -------
    pd.DataFrame
        Index (event_id, hit_id), one column per stage (chronological, signed values),
        plus `peak_stage` (stage of max |value|), `top_in` (stages where the hit was
        in the top-n) and `n_top` (len of `top_in`).
    """
    stages = list(df_dict)
    reference = df_dict[stages[0]]

    # Rows are aligned across stages, so a hit is identified by its row position.
    top_rows = {s: set(get_highest_values(df, n).index) for s, df in df_dict.items()}
    rows = np.array(sorted(set().union(*top_rows.values())), dtype=np.int64)

    wide = reference.iloc[rows][["event_id", "hit_id"]].copy()
    for s in stages:
        wide[s] = df_dict[s]["value"].to_numpy()[rows]
    wide.index = pd.MultiIndex.from_frame(wide[["event_id", "hit_id"]])
    wide = wide[stages]
    wide.columns.name = "stage"

    wide["peak_stage"] = wide[stages].abs().idxmax(axis=1)
    wide["top_in"] = [[s for s in stages if row in top_rows[s]] for row in rows]
    wide["n_top"] = wide["top_in"].str.len()

    wide = wide[wide["n_top"] >= min_stages]
    return wide.sort_values(stages[-1], ascending=False, key=lambda s: s.abs())


def attach_event_index(selection: pd.DataFrame, events_csv: Path):
    """Add root file and entry per selected hit from build_event_index.py output.

    Returns (selection, info). Joins only if events_meta.json next to the
    index reports check L1 as passed.
    """
    if not events_csv.is_file():
        return selection, {"joined": False, "reason": f"{events_csv} not found"}

    meta_path = events_csv.with_name("events_meta.json")
    l1_status = None
    if meta_path.is_file():
        with open(meta_path, encoding="utf-8") as handle:
            l1_status = json.load(handle).get("checks", {}).get("L1", {}).get("status")
    if l1_status != "passed":
        return selection, {
            "joined": False,
            "reason": f"check L1 is {l1_status!r} in {meta_path}, not 'passed'",
        }

    events = pd.read_csv(events_csv, usecols=["event_id"] + EVENT_INDEX_COLUMNS)
    joined = selection.merge(events, on="event_id", how="left", validate="many_to_one")
    if joined["root_file"].isna().any():
        missing = sorted(joined.loc[joined["root_file"].isna(), "event_id"].unique())
        raise ValueError(f"event_ids missing from {events_csv}: {missing[:10]}")
    return joined, {"joined": True, "path": str(events_csv)}


def colour_scale(stage_frames: dict, log_scale: float) -> dict:
    """|z| maximum and 99.9th percentile per stage and over all stages.

    z is the signed_log10 transform used in the histograms, so the viewer can
    use one colour scale for every event.
    """
    abs_g_max, abs_g_p999 = {}, {}
    for stage, frame in stage_frames.items():
        abs_g = np.abs(frame["value"].to_numpy())
        abs_g_max[stage] = float(abs_g.max())
        abs_g_p999[stage] = float(np.quantile(abs_g, 0.999))

    def to_z(g):
        return float(np.log10(1 + g / log_scale))

    return {
        "abs_z_max": {**{s: to_z(g) for s, g in abs_g_max.items()},
                      "all": to_z(max(abs_g_max.values()))},
        "abs_z_p999": {s: to_z(g) for s, g in abs_g_p999.items()},
    }


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------
def plot_top_hits_evolution(
    wide: pd.DataFrame, stage_cols: list, n: int, min_stages: int,
    labels: dict | None = None, save_dir=None, formats=("png", "pdf"), dpi=300,
):
    from matplotlib.ticker import FuncFormatter
    from matplotlib.lines import Line2D

    if wide.empty:
        print(f"No hit is in the top-{n} of >= {min_stages} stages.")
        return None

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(stage_cols))

    for (event_id, hit_id), row in wide.iterrows():
        vals = row[stage_cols].to_numpy(dtype=float)
        (line,) = ax.plot(
            x, np.abs(vals), linewidth=1.5, marker=None,
            label=f"ev {event_id} / hit {hit_id}  ({row['n_top']}×)",
        )
        c = line.get_color()
        pos, neg = vals > 0, vals < 0
        ax.scatter(x[pos], vals[pos], marker="^", s=45, color=c, zorder=3)
        ax.scatter(x[neg], -vals[neg], marker="v", s=45,
                   facecolors="none", edgecolors=c, linewidths=1.5, zorder=3)

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0e}"))
    ax.tick_params(axis="y", which="both", labelsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([(labels or {}).get(s, s) for s in stage_cols])
    ax.set_xlabel("stage")
    ax.set_ylabel("abs(gradient)")
    ax.set_title(rf"Top-{n} $\|$gradient$\|$ hits in ≥ {min_stages} stages ({len(wide)} hits)")

    sign_handles = [
        Line2D([], [], marker="^", color="gray", linestyle="none", markersize=8, label="positive"),
        Line2D([], [], marker="v", color="gray", markerfacecolor="none",
               linestyle="none", markersize=8, label="negative"),
    ]
    sign_legend = ax.legend(handles=sign_handles, title="±", loc="best", fontsize=10)
    ax.add_artist(sign_legend)  # keep it when the hit legend is added below

    if len(wide) < 20:
        ax.legend(
            title="event / hit  (# stages in top-n)", loc="upper left",
            bbox_to_anchor=(1.01, 1), fontsize=10, frameon=False,
        )
    fig.tight_layout()

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for fmt in formats:
            out_path = save_dir / f"top_hits_evolution.{fmt}"
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
            print(f"Saved {out_path}")
    plt.close(fig)
    return fig


def original_plot_gradient_histogram(df:pd.DataFrame, xlim=None):
    # log_10_value=np.log...
    Nbins, binwidth, bins, counts, bin_centers = getHistoParam(df["value"], Nbins=100)

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.hist(bin_centers, bins=bins, weights=counts, histtype='step', color=getColour(0), label="Gradient values")
    ax.set_xlabel("Gradient value")
    ax.set_ylabel("Counts")
    if xlim is not None:
        ax.set_xlim(xlim)

def signed_log10(values, scale=1e-6):
    values = np.asarray(values, dtype=np.float64)
    return np.sign(values) * np.log10(1 + np.abs(values) / scale)

def inverse_signed_log10(z, scale=1e-6):
    z = np.asarray(z, dtype=np.float64)
    return np.sign(z) * scale * (10**np.abs(z) - 1)

def latex_scientific(x):
    exponent = int(np.floor(np.log10(abs(x))))
    coefficient = x / 10**exponent

    if np.isclose(coefficient, 1):
        return rf"10^{{{exponent}}}"
    else:
        return rf"{coefficient:g}\times10^{{{exponent}}}"

def plot_gradient_histogram(
    df: pd.DataFrame,
    title=None,
    xlim=None,
    log_scale=1e-8,
    count_every=15,
    count_fontsize=12,
    count_rotation=0,
    count_min=1,
    count_y_offset_frac=0.015,
    ax=None,
    show_legend=True,
):
    log_values = signed_log10(df["value"], scale=log_scale)

    Nbins, binwidth, bins, counts, bin_centers = getHistoParam(
        log_values,
        Nbins=100
    )

    # NEW: use supplied axis if available
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 8))
    else:
        fig = ax.figure

    ax.hist(
        bin_centers,
        bins=bins,
        weights=counts,
        histtype="step",
        color=getColour(0),
        label="Gradient values",
        count_fontsize=count_fontsize,
        count_rotation=count_rotation,
        count_min=count_min,
        count_every=count_every,
        count_y_offset_frac=count_y_offset_frac,
    )

    scale_label = latex_scientific(log_scale)

    ax.set_xlabel(
        rf"Transformed gradient: "
        rf"$z=\mathrm{{sign}}(g)\log_{{10}}"
        rf"\left(1 + \frac{{|g|}}{{{scale_label}}}\right)$"
    )
    ax.set_ylabel("Counts")

    if xlim is not None:
        ax.set_xlim(xlim)

    if show_legend:
        ax.legend()

    if title is not None:
        ax.set_title(title)

    # -----------------------------------------------------------------
    # statistics
    # -----------------------------------------------------------------

    positive_counts = counts[bin_centers > 0]
    negative_counts = counts[bin_centers < 0]

    positive_peak_bin_centres = (
        bin_centers[bin_centers > 0][np.argmax(positive_counts)]
        if len(positive_counts) > 0 else None
    )
    negative_peak_bin_centres = (
        bin_centers[bin_centers < 0][np.argmax(negative_counts)]
        if len(negative_counts) > 0 else None
    )

    positive_peak_count = (
        np.max(positive_counts)
        if len(positive_counts) > 0 else None
    )
    negative_peak_count = (
        np.max(negative_counts)
        if len(negative_counts) > 0 else None
    )

    d_N = {
        "POPULATION": "",
        "N": f"{len(log_values):,}",
        "N+": f"{len(log_values[log_values > 0]):,}",
        "Npeak+": f"{positive_peak_count:,}",
        "Peak+": positive_peak_bin_centres,
        "N-": f"{len(log_values[log_values < 0]):,}",
        "Npeak-": f"{negative_peak_count:,}",
        "Peak-": negative_peak_bin_centres,
    }

    positive_values = log_values[log_values > 0]
    negative_values = log_values[log_values < 0]

    positive_median = (
        np.median(positive_values)
        if len(positive_values) > 0 else None
    )
    negative_median = (
        np.median(negative_values)
        if len(negative_values) > 0 else None
    )
    positive_mean = (
        np.mean(positive_values)
        if len(positive_values) > 0 else None
    )
    negative_mean = (
        np.mean(negative_values)
        if len(negative_values) > 0 else None
    )
    positive_std = (
        np.std(positive_values)
        if len(positive_values) > 0 else None
    )
    negative_std = (
        np.std(negative_values)
        if len(negative_values) > 0 else None
    )

    d_dist = {
        "TRANSFORMED": "",
        "max": np.max(log_values),
        "median+": positive_median,
        "mean+": positive_mean,
        "std+": positive_std,
        "min": np.min(log_values),
        "median-": negative_median,
        "mean-": negative_mean,
        "std-": negative_std,
    }

    positive_peak_gradient = inverse_signed_log10(
        positive_peak_bin_centres,
        scale=log_scale
    )
    negative_peak_gradient = inverse_signed_log10(
        negative_peak_bin_centres,
        scale=log_scale
    )

    # Raw gradients
    gradients = df["value"].to_numpy()
    g_pos = gradients[gradients > 0]
    g_neg = gradients[gradients < 0]

    d_grad = {
        "GRADIENTS": "",
        "max": f"{np.max(gradients):.3e}",
        "median+": f"{np.median(g_pos):.3e}",
        "peak+": f"{positive_peak_gradient:.3e}",
        "mean+": f"{np.mean(g_pos):.3e}",
        "std+": f"{np.std(g_pos):.3e}",
        "min": f"{np.min(gradients):.3e}",
        "median-": f"{np.median(g_neg):.3e}",
        "peak-": f"{negative_peak_gradient:.3e}",
        "mean-": f"{np.mean(g_neg):.3e}",
        "std-": f"{np.std(g_neg):.3e}",
    }

    left_text_upper_y_position = 0.75
    right_text_upper_y_position = 0.75

    add_text_to_ax(
        0.01,
        left_text_upper_y_position,
        nice_string_output(d_N),
        ax=ax,
        fontsize=12
    )

    add_text_to_ax(
        0.01,
        left_text_upper_y_position - 0.32,
        nice_string_output(d_dist),
        ax=ax,
        fontsize=12
    )

    add_text_to_ax(
        0.70,
        right_text_upper_y_position,
        nice_string_output(d_grad),
        ax=ax,
        fontsize=12
    )

    return fig, ax, d_N, d_dist, d_grad

def plot_all_gradient_histograms(
    dataframes,
    titles=None,
    ncols=3,
    figsize=None,
    sharex=True,
    sharey=True,
    **hist_kwargs,
):
    """
    Plot several gradient distributions in one figure.

    Parameters
    ----------
    dataframes : list of pd.DataFrame
        Gradient dataframes containing a 'value' column.

    titles : list of str, optional
        One title per dataframe.

    ncols : int
        Number of subplot columns.

    figsize : tuple, optional
        Figure size. Automatically determined if None.

    sharex, sharey : bool
        Share axes between panels.

    **hist_kwargs
        Additional arguments passed to plot_gradient_histogram().
    """

    nplots = len(dataframes)

    if titles is None:
        titles = [None] * nplots

    if len(titles) != nplots:
        raise ValueError(
            "Number of titles must match number of dataframes."
        )

    nrows = int(np.ceil(nplots / ncols))

    if figsize is None:
        figsize = (7 * ncols, 6 * nrows)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        sharex=sharex,
        sharey=sharey,
        squeeze=False,
    )

    axes_flat = axes.ravel()

    stats = {}

    for i, (df, title) in enumerate(zip(dataframes, titles)):

        _, ax, d_N, d_dist, d_grad = plot_gradient_histogram(
            df=df,
            title=title,
            ax=axes_flat[i],
            show_legend=(i == 0),
            **hist_kwargs,
        )

        stats[title if title is not None else i] = {
            "population": d_N,
            "transformed": d_dist,
            "gradient": d_grad,
        }

    # Remove unused axes
    for j in range(nplots, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    fig.tight_layout()


def plot_gradient_histograms_overlay(
    datasets,
    title=None,
    xlim=None,
    log_scale=1e-8,
    Nbins=100,
    density=False,
    annotate_counts=False,
    count_every=15,
    count_fontsize=10,
    count_rotation=0,
    count_min=1,
    count_y_offset_frac=0.015,
    save_dir=None,
    filename="gradient_histograms_overlay",
    formats=("png", "pdf"),
    dpi=300,
):
    """
    Overlay several transformed-gradient histograms on one axis.

    Parameters
    ----------
    datasets : dict
        Mapping from label -> dataframe.
        Each dataframe must contain a 'value' column.

        Example:
        {
            "@Initial Input": df_input,
            "@Initial Processing": df_initial,
            "@After GN1": df_post_gn1,
        }

    density : bool
        If False, plot raw counts.
        If True, normalise each histogram to a probability density.

    save_dir : str or Path, optional
        If given, the figure is saved there as `<filename>.<fmt>` for each
        entry in `formats`. The directory is created if needed.
    """

    # ------------------------------------------------------------
    # Transform all distributions first
    # ------------------------------------------------------------
    transformed = {}

    for label, df in datasets.items():
        z = signed_log10(df["value"], scale=log_scale)
        z = np.asarray(z)
        z = z[np.isfinite(z)]

        transformed[label] = z

    # ------------------------------------------------------------
    # Define ONE common set of bins from all distributions
    # ------------------------------------------------------------
    all_values = np.concatenate(list(transformed.values()))

    if xlim is None:
        _, _, bins, _, _ = getHistoParam(
            all_values,
            Nbins=Nbins
        )
    else:
        # If a visible range is explicitly requested, make the bins
        # directly over that range.
        bins = np.linspace(
            xlim[0],
            xlim[1],
            Nbins + 1
        )

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(15, 10))

    for i, (label, z) in enumerate(transformed.items()):

        ax.hist(
            z,
            bins=bins,
            density=density,
            histtype="step",
            linewidth=2,
            color=getColour(i),
            label=label,

            # controls from your patched Axes.hist
            annotate_counts=annotate_counts,
            count_fontsize=count_fontsize,
            count_rotation=count_rotation,
            count_min=count_min,
            count_every=count_every,
            count_y_offset_frac=count_y_offset_frac,
        )

    # ------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------
    scale_label = latex_scientific(log_scale)

    ax.set_xlabel(
        rf"Transformed gradient: "
        rf"$z=\mathrm{{sign}}(g)\log_{{10}}"
        rf"\left(1+\frac{{|g|}}{{{scale_label}}}\right)$"
    )

    ax.set_ylabel(
        "Probability density" if density else "Counts"
    )

    if xlim is not None:
        ax.set_xlim(xlim)

    if title is not None:
        ax.set_title(title)

    ax.legend()

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for fmt in formats:
            out_path = save_dir / f"{filename}.{fmt}"
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
            print(f"Saved {out_path}")

    return fig, ax


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
def run(args):
    stage_files = discover_stages(args.probe_dir)
    print("Stages:", ", ".join(f"{k} ({p.name})" for k, p in stage_files.items()))

    stage_frames = {}
    for stage, path in stage_files.items():
        print(f"  reading {path.name}")
        stage_frames[stage] = read_probe_file(path)
    check_aligned(stage_frames)
    first = next(iter(stage_frames.values()))
    n_events = int(first["event_id"].nunique())
    print(f"{len(first):,} hits in {n_events:,} events, identical across "
          f"{len(stage_frames)} stages.")

    labels = {s: STAGE_LABELS.get(s, s) for s in stage_frames}

    # Distribution of all stages
    fig, _ = plot_gradient_histograms_overlay(
        {labels[s]: frame for s, frame in stage_frames.items()},
        title="Gradient distribution across probing locations",
        log_scale=args.log_scale,
        Nbins=100,
        save_dir=args.out_dir / "figures",
    )
    plt.close(fig)

    # Selection
    selection_name = args.selection_name or f"top{args.top_n}_min{args.min_stages}"
    selection_dir = args.out_dir / "selections" / selection_name
    stages = list(stage_frames)

    wide = track_top_hits(stage_frames, n=args.top_n, min_stages=args.min_stages)
    plot_top_hits_evolution(
        wide, stages, n=args.top_n, min_stages=args.min_stages,
        labels=labels, save_dir=selection_dir,
    )

    selection = wide.reset_index()
    selection["top_in"] = selection["top_in"].str.join(";")
    events_csv = args.events_csv or (args.out_dir / "events.csv")
    selection, index_info = attach_event_index(selection, Path(events_csv))
    if not index_info["joined"]:
        print(f"WARNING: no root file/entry in the selection: {index_info['reason']}")

    selection_dir.mkdir(parents=True, exist_ok=True)
    selection_path = selection_dir / "selection.csv"
    selection.to_csv(selection_path, index=False)

    meta = {
        "script": Path(__file__).name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "probe_dir": str(args.probe_dir),
        "stages": [
            {"key": s, "label": labels[s], "file": str(stage_files[s]),
             "n_rows": int(len(stage_frames[s]))}
            for s in stages
        ],
        "n_hits": int(len(first)),
        "n_events": n_events,
        "criteria": {
            "rule": ("per stage: hit with max |value| in each event, then the "
                     "top_n events; union over stages, kept if in >= min_stages"),
            "top_n": args.top_n,
            "min_stages": args.min_stages,
        },
        "n_selected_hits": int(len(selection)),
        "n_selected_events": int(selection["event_id"].nunique()),
        "transform": {
            "name": "signed_log10",
            "formula": "z = sign(g) * log10(1 + |g| / scale)",
            "scale": args.log_scale,
        },
        "colour_scale": colour_scale(stage_frames, args.log_scale),
        "event_index": index_info,
    }
    with open(selection_dir / "selection_meta.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    print(f"\nSelected {meta['n_selected_hits']} hits in "
          f"{meta['n_selected_events']} events.")
    print(f"Wrote {selection_path}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Gradient distributions and hit selection for one probe run.",
    )
    parser.add_argument("--probe-dir", required=True, type=Path,
                        help="directory with one CSV per probe stage")
    parser.add_argument("--out-dir", required=True, type=Path,
                        help="run directory on /ceph, e.g. /ceph/cjo/GradientProbe/runs/<run>")
    parser.add_argument("--events-csv", type=Path, default=None,
                        help="build_event_index.py output (default: <out-dir>/events.csv)")
    parser.add_argument("--top-n", type=int, default=20,
                        help="events taken per stage (default 20)")
    parser.add_argument("--min-stages", type=int, default=1,
                        help="keep hits in the top-n of at least this many stages (default 1)")
    parser.add_argument("--log-scale", type=float, default=1e-8,
                        help="scale of the signed_log10 transform (default 1e-8)")
    parser.add_argument("--selection-name", default=None,
                        help="sub-directory name (default top<n>_min<m>)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    # plot style settings
    setMplParam()
    enable_histogram_counts()
    run(args)


if __name__ == "__main__":
    main()