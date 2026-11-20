"""
beta_statistics.py

Purpose
-------
Distribution of the condensation score beta of one run, and how it is
spread over the per-hit probe gradients.

beta is computed once per hit, at the output (sigmoid(linear_beta) after the
output head); there is no beta at the intermediate probe stages. The
gradients g, however, are measured at every probe stage, so beta is compared
with each of them:

  * Histogram of beta over all hits (BETA_NBINS = 50 bins), with a
    statistics box, in the style of the gradient histograms.
  * Scatter of every hit's (z, beta), one figure per stage, z on the x axis:
    z = sign(g) log10(1 + |g|/scale) is the transform and scale of
    gradient_statistics.py. Marginal histograms: beta on the left (rotated,
    sharing the beta axis; the same in every figure), z of that stage below
    (Z_NBINS bins). Scatter and marginals use the same random N_SCATTER =
    1,000,000 hits (all hits if fewer), identical at every stage, so all
    figures have the same axes and show the same hits. The corner gives the
    hit counts (shown, z > 0, z < 0).
  * The same scatters (without marginals) of all stages in one figure, one
    panel per stage, shared axes and one common z axis label.
  * The (z, beta) population density of all stages in one figure: a 2D
    histogram per stage over the same N_SCATTER hits as the scatters,
    DENSITY_NBINS x DENSITY_NBINS bins, z in [-lim, lim] with lim = max |z|
    over all stages and these hits (or --zlim), beta in [0, 1]. Shared bins, axes
    and colour scale (YlGnBu, log counts unless --linear-counts), one colour
    bar; empty bins stay blank.
  * The same density figure with the raw gradient g instead of z: linear
    g bins over [-glim, glim], glim = max |g| over all stages and the same
    N_SCATTER hits (or --glim), same beta bins, colour map and colour scale rules. It shows
    what the transform hides: how strongly g piles up near zero, and how
    the gradient scale differs between stages.

Inputs: the stage CSVs in --probe-dir and the beta CSV of the same dump
(--beta-file; without it, beta.csv in --probe-dir or its parent, as in
gradient_statistics.py). Rows of the beta CSV must align with the stage CSVs.

Outputs (in --out-dir/beta, --out-dir = the run directory)
----------------------------------------------------------
beta_histogram.{png,pdf}
z_vs_beta.{png,pdf}                         all stages, one scatter panel each
z_vs_beta/<i>_<stage>.{png,pdf}             one figure per stage, i = network order
z_vs_beta_density.pdf                       all stages, one 2D histogram panel each
z_vs_beta_density_counts.csv                the counts of every bin (stage, bin edges, count)
g_vs_beta_density.pdf                       the same with the raw gradient g on x
g_vs_beta_density_counts.csv                its bin counts
beta_summary.csv                            per stage: hits drawn, N+, N-
beta_meta.json

cyan.jo, Claude Opus5.5 Extra, 2026-11-19
cyan.jo, Claude Opus5.5 Extra, 2026-11-19: --beta-file; bin number fixed in
the script (BETA_NBINS) instead of --bins; one (z, beta) figure per stage
instead of one grid figure; no correlation coefficient (distribution only);
marginal histograms of beta (left) and z (bottom) around each scatter;
always N_SCATTER = 1,000,000 random hits in the scatter figures (no N0);
all-stage figure z_vs_beta (scatter panels only) next to the per-stage ones.
cyan.jo, Claude Opus5.5 Extra, 2026-11-20: all-stage (z, beta) density
figure z_vs_beta_density (2D histograms, DENSITY_NBINS bins per axis,
YlGnBu) and its bin counts; --linear-counts; --zlim also sets its z range.
cyan.jo, Claude Opus5.5 Extra, 2026-11-20: raw-gradient density figure
g_vs_beta_density and its bin counts (--glim); the density plotting is shared
by the z and g figures (plot_density_grid). Both density figures use the
scatter's N_SCATTER subsample instead of all hits.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import FuncFormatter, MaxNLocator

# Utils (PlotUtils, ExternalFunctions) is put on PYTHONPATH by run_beta_statistics.sh.
try:
    from PlotUtils import setMplParam, getColour, getHistoParam, enable_histogram_counts
    from ExternalFunctions import add_text_to_ax, nice_string_output
except ModuleNotFoundError as error:
    raise SystemExit(
        f"{error}. PlotUtils and ExternalFunctions live in /work/cjo/Utils; run this "
        "script through run_beta_statistics.sh, which puts that folder on PYTHONPATH."
    ) from error
# Shared with the statistics step: same stage discovery, checks and transform.
from gradient_statistics import (STAGE_LABELS,
                                 check_aligned,
                                 discover_stages,
                                 find_dump_file,
                                 latex_scientific,
                                 read_dump_meta,
                                 read_probe_file,
                                 signed_log10)


# Bins of the beta histogram. getHistoParam spans [floor(min), ceil(max)] = [0, 1]
# for beta, so 50 bins are 0.02 wide.
BETA_NBINS = 50
# Bins of the z histograms, as in gradient_statistics.py (Nbins=100).
Z_NBINS = 100
# Hits in every (z, beta) figure: a random subsample, the same at every stage.
N_SCATTER = 1_000_000
# Bins per axis of the (z, beta) density figure: z in [-lim, lim] and beta in
# [0, 1], each in DENSITY_NBINS bins (50 bins: lim = 7.5 gives z bins 0.3 wide,
# beta bins 0.02). The raw-gradient density figure uses the same number.
DENSITY_NBINS = 50
# Colour map of the density figure.
DENSITY_CMAP = "YlGnBu"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def z_label(log_scale) -> str:
    return (rf"Transformed gradient: $z=\mathrm{{sign}}(g)\log_{{10}}"
            rf"\left(1+\frac{{|g|}}{{{latex_scientific(log_scale)}}}\right)$")


def compact_count(value, _position=None) -> str:
    """Tick label for counts without an offset text: 300k, 1.2M."""
    for factor, suffix in ((1e6, "M"), (1e3, "k")):
        if abs(value) >= factor:
            return f"{value / factor:g}{suffix}"
    return f"{value:g}"


def save_figure(fig, save_dir, filename, formats=("pdf",), dpi=300):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        out_path = save_dir / f"{filename}.{fmt}"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved {out_path}")


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------
def plot_beta_histogram(
    beta,
    *,
    n_events,
    Nbins=BETA_NBINS,
    xlim=(-0.05, 1.05),
    beta_threshold=None,
    count_every=1,
    count_fontsize=9,
    count_rotation=90,
    count_min=1,
    count_y_offset_frac=0.015,
    ax=None,
):
    """Step histogram of beta over all hits, with counts and a statistics box.

    Returns (fig, ax, stats).
    """
    beta = np.asarray(beta, dtype=np.float64)
    Nbins, binwidth, bins, counts, bin_centers = getHistoParam(beta, Nbins=Nbins)

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
        label=r"$\beta$",
        count_fontsize=count_fontsize,
        count_rotation=count_rotation,
        count_min=count_min,
        count_every=count_every,
        count_y_offset_frac=count_y_offset_frac,
    )
    # Headroom for the count labels: the patched hist clips them at the y limit.
    ax.set_ylim(0, 1.15 * counts.max())
    if beta_threshold is not None:
        ax.axvline(beta_threshold, color="gray", linestyle=":", linewidth=1.5,
                   label=rf"$t_\beta$ = {beta_threshold:g}")

    ax.set_title(rf"Histogram of $\beta$ values ({len(beta):,} hits in {n_events:,} events)")
    ax.set_xlabel(r"$\beta$ value")
    ax.set_ylabel("Counts")
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.legend(loc="upper right")

    stats = {
        "BETA": "",
        "N": f"{len(beta):,}",
        "bin width": f"{binwidth:.4f}",
        "min": f"{beta.min():.4f}",
        "max": f"{beta.max():.4f}",
        "mean": f"{beta.mean():.4f}",
        "median": f"{np.median(beta):.4f}",
        "std": f"{beta.std():.4f}",
    }
    if beta_threshold is not None:
        stats[f"frac > {beta_threshold:g}"] = f"{(beta > beta_threshold).mean():.4f}"
    add_text_to_ax(0.62, 0.85, nice_string_output(stats), ax=ax, fontsize=12)
    return fig, ax, stats


def hit_counts(z) -> dict:
    """Hit counts of one stage: shown, z > 0, z < 0."""
    return {
        "N": f"{len(z):,}",
        "N+": f"{int((z > 0).sum()):,}",
        "N-": f"{int((z < 0).sum()):,}",
    }


def draw_scatter(ax, z, beta, *, colour, zlim, beta_threshold=None,
                 marker_size=1, alpha=0.1):
    """(z, beta) of every hit on ax: z on x, beta on y, rasterized points."""
    ax.scatter(z, beta, s=marker_size, alpha=alpha, color=colour,
               linewidths=0, rasterized=True)
    ax.axvline(0, color="gray", linewidth=0.8)
    if beta_threshold is not None:
        ax.axhline(beta_threshold, color="gray", linestyle=":", linewidth=1.5)
    ax.set_xlim(-zlim, zlim)
    ax.set_ylim(0, 1)


def plot_z_vs_beta(
    z,
    beta,
    *,
    title=None,
    zlim=None,
    log_scale=1e-8,
    colour=None,
    beta_colour=None,
    beta_threshold=None,
    marker_size=1,
    alpha=0.1,
    z_nbins=Z_NBINS,
    beta_nbins=BETA_NBINS,
):
    """Scatter of every hit's (z, beta) at one stage with both marginal histograms.

    Layout (2 x 2 grid):
        [ beta histogram ] [ scatter, z on x, beta on y ]
        [ hit counts     ] [ z histogram                 ]
    The beta histogram shares the beta axis with the scatter and grows to the
    left, so it keeps the vertical beta axis; the z histogram shares the z
    axis and is upright (counts grow upwards).
    Scatter points are rasterized, so the pdf stays small with ~1e6 hits.
    Returns (fig, axes dict, stats); stats holds the hit counts.
    """
    z = np.asarray(z, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    colour = colour or getColour(0)
    beta_colour = beta_colour or getColour(0)
    if zlim is None:
        zlim = 1.05 * float(np.abs(z).max()) or 1.0

    fig = plt.figure(figsize=(14, 12))
    grid = fig.add_gridspec(2, 2, width_ratios=(1.3, 4), height_ratios=(4, 1.2),
                            wspace=0.05, hspace=0.05)
    ax = fig.add_subplot(grid[0, 1])
    ax_beta = fig.add_subplot(grid[0, 0], sharey=ax)
    ax_z = fig.add_subplot(grid[1, 1], sharex=ax)
    ax_text = fig.add_subplot(grid[1, 0])
    ax_text.axis("off")

    # Centre: the scatter.
    draw_scatter(ax, z, beta, colour=colour, zlim=zlim, beta_threshold=beta_threshold,
                 marker_size=marker_size, alpha=alpha)
    ax.tick_params(labelbottom=False, labelleft=False)
    if title is not None:
        ax.set_title(title)

    # Left: beta histogram, horizontal, growing to the left. Same bins as
    # the beta histogram figure ([0, 1] in BETA_NBINS bins).
    _, _, beta_bins, _, _ = getHistoParam(beta, Nbins=beta_nbins)
    ax_beta.hist(beta, bins=beta_bins, orientation="horizontal", histtype="step",
                 color=beta_colour, annotate_counts=False)
    if beta_threshold is not None:
        ax_beta.axhline(beta_threshold, color="gray", linestyle=":", linewidth=1.5)
    ax_beta.invert_xaxis()
    ax_beta.set_ylabel(r"$\beta$ value")
    # Count axis on top: the bottom-left corner holds the hit counts.
    ax_beta.xaxis.tick_top()
    ax_beta.xaxis.set_label_position("top")
    ax_beta.set_xlabel("Counts")
    ax_beta.xaxis.set_major_locator(MaxNLocator(3))
    ax_beta.xaxis.set_major_formatter(FuncFormatter(compact_count))

    # Bottom: z histogram, upright, bins over the shown z range.
    z_bins = np.linspace(-zlim, zlim, z_nbins + 1)
    ax_z.hist(z, bins=z_bins, histtype="step", color=colour, annotate_counts=False)
    ax_z.axvline(0, color="gray", linewidth=0.8)
    ax_z.set_xlabel(z_label(log_scale))
    ax_z.set_ylabel("Counts")
    ax_z.yaxis.set_major_locator(MaxNLocator(3))
    ax_z.yaxis.set_major_formatter(FuncFormatter(compact_count))

    # Bottom left: hit counts.
    stats = hit_counts(z)
    add_text_to_ax(0.0, 0.75, nice_string_output(stats), ax=ax_text, fontsize=12)
    return fig, {"scatter": ax, "beta": ax_beta, "z": ax_z}, stats


def plot_z_vs_beta_grid(
    z_by_stage,
    beta,
    *,
    labels,
    zlim,
    log_scale=1e-8,
    beta_threshold=None,
    marker_size=1,
    alpha=0.1,
    ncols=3,
):
    """The scatter of every stage in one figure: one panel per stage, network
    order, shared axes, one z axis label for the whole figure, hit counts in
    each panel. Colours as in the per-stage figures (getColour(i)).
    Returns (fig, axes).
    """
    stages = list(z_by_stage)
    nrows = int(np.ceil(len(stages) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    for i, (ax, stage) in enumerate(zip(axes.ravel(), stages)):
        draw_scatter(ax, z_by_stage[stage], beta, colour=getColour(0), zlim=zlim,
                     beta_threshold=beta_threshold, marker_size=marker_size, alpha=alpha)
        ax.set_title(labels[stage])
        add_text_to_ax(0.02, 0.97, nice_string_output(hit_counts(z_by_stage[stage])),
                       ax=ax, fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\beta$ value")
    for ax in axes.ravel()[len(stages):]:
        fig.delaxes(ax)

    # One z label below all panels instead of one per column.
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.supxlabel(z_label(log_scale), y=0.01,
                  fontsize=plt.rcParams["axes.labelsize"])
    return fig, axes


def symmetric_edges(values_by_stage, *, nbins=DENSITY_NBINS, lim=None):
    """nbins equal bins over [-lim, lim], shared by every stage.

    lim = the given lim, else the largest |value| over all stages, so the
    most extreme hit sits in an edge bin.
    """
    if lim is None:
        lim = max(float(np.abs(v).max()) for v in values_by_stage.values()) or 1.0
    return np.linspace(-lim, lim, nbins + 1)


def beta_bin_edges(nbins=DENSITY_NBINS):
    """nbins equal beta bins over [0, 1]."""
    return np.linspace(0.0, 1.0, nbins + 1)


def plot_density_grid(
    x_by_stage,
    beta,
    *,
    labels,
    x_edges,
    beta_edges,
    xlabel,
    x_symbol,
    log_counts=True,
    beta_threshold=None,
    cmap=DENSITY_CMAP,
    scientific_x=False,
    extra_text=None,
    ncols=3,
):
    """Population density of (x, beta) at every stage in one figure.

    x is z (transformed gradient) or g (raw gradient); beta on y as in
    z_vs_beta. One 2D histogram panel per stage, network order. All panels
    share the bin edges, the axes and the colour scale (log counts from 1 to
    the largest bin of any stage, or linear from 0), with one colour bar.
    Empty bins are masked and show the background, so "no hit" is told
    apart from "few hits". Hits outside the x range (only with a fixed
    range below the largest |x|) are not drawn; the panel says how many.
    x_symbol is the LaTeX symbol of x in the title (e.g. "z", "g").
    scientific_x writes the x ticks as multiples of a power of ten (raw g).
    extra_text: optional {stage: {key: text}} added to each panel's counts.
    Returns (fig, axes, counts); counts[stage] has shape
    (len(x_edges) - 1, len(beta_edges) - 1), indexed [x bin, beta bin].
    """
    beta = np.asarray(beta, dtype=np.float64)
    counts = {stage: np.histogram2d(np.asarray(x, dtype=np.float64), beta,
                                    bins=(x_edges, beta_edges))[0]
              for stage, x in x_by_stage.items()}
    vmax = max(1.0, max(float(c.max()) for c in counts.values()))
    norm = LogNorm(vmin=1, vmax=vmax) if log_counts else Normalize(vmin=0, vmax=vmax)

    stages = list(x_by_stage)
    nrows = int(np.ceil(len(stages) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows),
                             sharex=True, sharey=True, squeeze=False,
                             constrained_layout=True)
    mesh = None
    for ax, stage in zip(axes.ravel(), stages):
        x = x_by_stage[stage]
        # pcolormesh takes C as [row = y bin, column = x bin]: transpose.
        mesh = ax.pcolormesh(x_edges, beta_edges,
                             np.ma.masked_equal(counts[stage].T, 0),
                             cmap=cmap, norm=norm, shading="flat", rasterized=True)
        ax.axvline(0, color="gray", linewidth=0.8)
        if beta_threshold is not None:
            ax.axhline(beta_threshold, color="gray", linestyle=":", linewidth=1.5)
        ax.set_xlim(x_edges[0], x_edges[-1])
        ax.set_ylim(beta_edges[0], beta_edges[-1])
        if scientific_x:
            ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0),
                                useMathText=True)
        ax.set_title(labels[stage])
        text = hit_counts(x)
        n_outside = len(x) - int(counts[stage].sum())
        if n_outside:
            text["outside"] = f"{n_outside:,}"
        if extra_text and stage in extra_text:
            text.update(extra_text[stage])
        add_text_to_ax(0.02, 0.97, nice_string_output(text), ax=ax, fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\beta$ value")
    for ax in axes.ravel()[len(stages):]:
        fig.delaxes(ax)

    shown = [ax for ax in axes.ravel()[:len(stages)]]
    colourbar = fig.colorbar(mesh, ax=shown, shrink=0.9, pad=0.01)
    colourbar.set_label("Hits per bin" + (" (log scale)" if log_counts else ""))

    nx, nb = len(x_edges) - 1, len(beta_edges) - 1
    fig.suptitle(
        rf"Population density of $({x_symbol}, \beta)$: {nx} $\times$ {nb} bins, "
        rf"$\Delta {x_symbol}$ = {x_edges[1] - x_edges[0]:.3g}, "
        rf"$\Delta\beta$ = {beta_edges[1] - beta_edges[0]:.3g}")
    fig.supxlabel(xlabel, fontsize=plt.rcParams["axes.labelsize"])
    return fig, axes, counts


def density_counts_table(counts, labels, x_edges, beta_edges, *, x_name="z") -> pd.DataFrame:
    """Long table of a density figure: one row per stage and bin, empty bins
    included; x columns are <x_name>_low, <x_name>_high."""
    xi, bi = np.meshgrid(np.arange(len(x_edges) - 1), np.arange(len(beta_edges) - 1),
                         indexing="ij")
    xi, bi = xi.ravel(), bi.ravel()
    frames = [pd.DataFrame({
        "stage": stage,
        "label": labels[stage],
        f"{x_name}_low": x_edges[xi], f"{x_name}_high": x_edges[xi + 1],
        "beta_low": beta_edges[bi], "beta_high": beta_edges[bi + 1],
        "count": c[xi, bi].astype(np.int64),
    }) for stage, c in counts.items()]
    return pd.concat(frames, ignore_index=True)


def density_meta(counts, x_edges, beta_edges, *, n_points, linear_counts) -> dict:
    """Settings and result size of one density figure, for beta_meta.json."""
    return {"nbins": len(x_edges) - 1, "n_points": int(n_points),
            "lim": float(x_edges[-1]),
            "x_bin_width": float(x_edges[1] - x_edges[0]),
            "beta_bin_width": float(beta_edges[1] - beta_edges[0]),
            "cmap": DENSITY_CMAP,
            "colour_scale": "linear" if linear_counts else "log",
            "max_count": int(max(c.max() for c in counts.values()))}


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
def run(args):
    stage_files = discover_stages(args.probe_dir, only=args.stages)
    dump_meta = read_dump_meta(args.probe_dir)
    print("Stages:", ", ".join(f"{k} ({p.name})" for k, p in stage_files.items()))

    if args.beta_file is not None:
        beta_file = args.beta_file
        if not beta_file.is_file():
            raise FileNotFoundError(f"--beta-file {beta_file} not found")
    else:
        beta_file = find_dump_file(args.probe_dir, "beta.csv")
        if beta_file is None:
            raise FileNotFoundError(
                f"No beta.csv in {args.probe_dir} or its parent; pass --beta-file.")

    stage_frames = {}
    for stage, path in stage_files.items():
        print(f"  reading {path.name}")
        stage_frames[stage] = read_probe_file(path)
    beta_frame = read_probe_file(beta_file)
    print(f"  beta from {beta_file}")
    first_stage = next(iter(stage_frames))
    check_aligned(stage_frames)
    check_aligned({first_stage: stage_frames[first_stage], "beta": beta_frame})

    beta = beta_frame["value"].to_numpy(dtype=np.float64)
    n_events = int(beta_frame["event_id"].nunique())
    print(f"{len(beta):,} hits in {n_events:,} events.")
    bad = int(((beta < 0) | (beta > 1) | ~np.isfinite(beta)).sum())
    if bad:
        print(f"WARNING: {bad} beta values outside [0, 1] or not finite.")

    labels = {s: STAGE_LABELS.get(s, s) for s in stage_frames}
    out_dir = args.out_dir / "beta"

    # 1. Histogram of beta
    fig, _, hist_stats = plot_beta_histogram(
        beta, n_events=n_events, Nbins=BETA_NBINS, beta_threshold=args.beta_threshold,
        count_every=args.count_every, count_rotation=args.count_rotation,
    )
    save_figure(fig, out_dir, "beta_histogram", formats=("pdf",))
    plt.close(fig)

    # 2. (z, beta) per stage: N_SCATTER random hits, the same at every stage.
    rows = np.arange(len(beta))
    if len(beta) > N_SCATTER:
        rows = np.sort(np.random.default_rng(args.seed).choice(
            len(beta), size=N_SCATTER, replace=False))
    print(f"Scatter: {len(rows):,} of {len(beta):,} hits (seed {args.seed}), "
          "the same hits at every stage.")
    z_by_stage = {s: signed_log10(f["value"].to_numpy()[rows], scale=args.log_scale)
                  for s, f in stage_frames.items()}
    zlim = args.zlim or 1.05 * max(float(np.abs(z).max()) for z in z_by_stage.values())
    scatter_kwargs = dict(zlim=zlim, log_scale=args.log_scale,
                          beta_threshold=args.beta_threshold,
                          marker_size=args.marker_size, alpha=args.alpha)

    # One figure per stage, numbered in network order.
    scatter_stats = {}
    for i, (stage, z) in enumerate(z_by_stage.items()):
        fig, _, scatter_stats[stage] = plot_z_vs_beta(
            z, beta[rows], title=rf"$z$ vs $\beta$ {labels[stage]}",
            colour=getColour(i % 10), beta_colour=getColour(0), **scatter_kwargs)
        save_figure(fig, out_dir / "z_vs_beta", f"{i + 1}_{stage}")
        plt.close(fig)

    # All stages in one figure: the same hits, axes and colours.
    fig, _ = plot_z_vs_beta_grid(
        z_by_stage, beta[rows], labels=labels, **scatter_kwargs)
    save_figure(fig, out_dir, "z_vs_beta")
    plt.close(fig)

    # 3. Population density over the scatter's N_SCATTER hits (the same rows
    #    at every stage and in both figures), DENSITY_NBINS bins per axis,
    #    shared edges and colour scale: (z, beta), then the raw gradient (g, beta).
    beta_shown = beta[rows]
    g_shown_by_stage = {s: f["value"].to_numpy(dtype=np.float64)[rows]
                        for s, f in stage_frames.items()}
    beta_edges = beta_bin_edges(DENSITY_NBINS)
    density_info = {}
    density_kwargs = dict(labels=labels, beta_edges=beta_edges,
                          log_counts=not args.linear_counts,
                          beta_threshold=args.beta_threshold)

    # z_by_stage holds the same rows, transformed (section 2).
    z_edges = symmetric_edges(z_by_stage, nbins=DENSITY_NBINS, lim=args.zlim)
    print(f"Density (z): {len(rows):,} of {len(beta):,} hits (the scatter's), {DENSITY_NBINS} x {DENSITY_NBINS} bins, "
          f"z in [{z_edges[0]:.3f}, {z_edges[-1]:.3f}] (bin width "
          f"{z_edges[1] - z_edges[0]:.3f}), beta bin width {beta_edges[1] - beta_edges[0]:.3f}.")
    fig, _, counts = plot_density_grid(
        z_by_stage, beta_shown, x_edges=z_edges, xlabel=z_label(args.log_scale),
        x_symbol="z", **density_kwargs)
    save_figure(fig, out_dir, "z_vs_beta_density")
    plt.close(fig)
    density_counts_table(counts, labels, z_edges, beta_edges, x_name="z").to_csv(
        out_dir / "z_vs_beta_density_counts.csv", index=False)
    density_info["z"] = density_meta(counts, z_edges, beta_edges, n_points=len(rows),
                                     linear_counts=args.linear_counts)

    g_edges = symmetric_edges(g_shown_by_stage, nbins=DENSITY_NBINS, lim=args.glim)
    print(f"Density (g): g in [{g_edges[0]:.3e}, {g_edges[-1]:.3e}] "
          f"(bin width {g_edges[1] - g_edges[0]:.3e}).")
    max_abs_g = {s: {"max |g|": f"{float(np.abs(g).max()):.2e}"}
                 for s, g in g_shown_by_stage.items()}
    fig, _, counts = plot_density_grid(
        g_shown_by_stage, beta_shown, x_edges=g_edges,
        xlabel=r"Raw gradient: $g = \partial L_{\mathrm{OC}} / \partial m$",
        x_symbol="g", scientific_x=True, extra_text=max_abs_g, **density_kwargs)
    save_figure(fig, out_dir, "g_vs_beta_density")
    plt.close(fig)
    density_counts_table(counts, labels, g_edges, beta_edges, x_name="g").to_csv(
        out_dir / "g_vs_beta_density_counts.csv", index=False)
    density_info["g"] = density_meta(counts, g_edges, beta_edges, n_points=len(rows),
                                     linear_counts=args.linear_counts)
    del g_shown_by_stage

    summary = pd.DataFrame([
        {"stage": s, "label": labels[s], **{k: v.replace(",", "") for k, v in st.items()}}
        for s, st in scatter_stats.items()])
    summary.to_csv(out_dir / "beta_summary.csv", index=False)
    print("\n" + summary.to_string(index=False))

    meta = {
        "script": Path(__file__).name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "probe_dir": str(args.probe_dir),
        "beta_file": str(beta_file),
        "stages": [{"key": s, "label": labels[s], "file": str(stage_files[s])}
                   for s in stage_frames],
        "n_hits": int(len(beta)),
        "n_events": n_events,
        "histogram": {"bins": BETA_NBINS, **hist_stats},
        "scatter": {"z_bins": Z_NBINS, "n_points": int(len(rows)),
                    "n_scatter": N_SCATTER, "seed": args.seed, "zlim": zlim},
        "density": density_info,
        "beta_threshold": args.beta_threshold,
        "transform": {"name": "signed_log10",
                      "formula": "z = sign(g) * log10(1 + |g| / scale)",
                      "scale": args.log_scale},
        "dump": ({k: dump_meta.get(k) for k in
                  ("_path", "checkpoint", "epoch", "val_loss_at_selection",
                   "bn_frozen", "locations")}
                 if dump_meta else None),
    }
    with open(out_dir / "beta_meta.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="beta histogram and (z, beta) scatter per probe stage for one run.",
    )
    parser.add_argument("--probe-dir", required=True, type=Path,
                        help="directory with one CSV per probe stage (beta.csv there or in its parent)")
    parser.add_argument("--beta-file", type=Path, default=None,
                        help="per-hit beta CSV (event_id,hit_id,value) of the same dump "
                             "(default: beta.csv in --probe-dir or its parent)")
    parser.add_argument("--out-dir", required=True, type=Path,
                        help="run directory on /ceph, e.g. /ceph/cjo/GradientProbe/runs/<run>")
    parser.add_argument("--stages", nargs="+", default=None,
                        help="probe locations to plot, e.g. post_initial post_gn1 post_gn2 "
                             "post_gn3 post_gn4 post_output_head (default: every stage)")
    parser.add_argument("--count-every", type=int, default=1,
                        help="count labels on the histogram: passed to the patched hist (default 1)")
    parser.add_argument("--count-rotation", type=float, default=90,
                        help="rotation of the count labels (default 90, readable with 50 bins)")
    parser.add_argument("--beta-threshold", type=float, default=None,
                        help="draw t_beta as a line and count beta above it (default: none)")
    parser.add_argument("--log-scale", type=float, default=1e-8,
                        help="scale of the signed_log10 transform (default 1e-8, as in "
                             "gradient_statistics.py)")
    parser.add_argument("--zlim", type=float, default=None,
                        help="z range [-zlim, zlim] of the scatter and density figures "
                             "(default: scatter 1.05 x max |z|, density max |z|, both of "
                             "the shown hits)")
    parser.add_argument("--glim", type=float, default=None,
                        help="g range [-glim, glim] of the raw-gradient density figure "
                             "(default: max |g| of the shown hits over all stages)")
    parser.add_argument("--linear-counts", action="store_true",
                        help="linear colour scale in the density figures (default: log)")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed of the N_SCATTER subsample of the scatter and density figures (default 0)")
    parser.add_argument("--marker-size", type=float, default=1.0,
                        help="scatter marker size in points^2 (default 1)")
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="scatter marker opacity (default 0.1)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    # plot style settings
    setMplParam()
    enable_histogram_counts()
    run(args)


if __name__ == "__main__":
    main()
