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
  * Zone tracking: boxes in the (z, beta) plane of one reference stage
    (ZONES, REF_STAGE) mark groups of hits; each zone keeps its colour
    (getColour(i), i = position in ZONES) at every stage, over a grey
    background of the shown hits that are in no zone. beta is the same for a hit at every
    stage, so a zone's hits can only move along z. Two figures: the (z, beta)
    scatter of every stage with the zones coloured (boxes drawn on the
    reference panel), and z at the reference stage against z at every other
    stage (diagonal = unchanged, anti-diagonal quadrants = sign flipped).
    The zone boxes are drawn on every panel of the (z, beta) figure (dashed
    on the reference, dotted elsewhere); next to each box (ZONE_TEXT_POS)
    the hits inside it at that stage, one line per reference zone in that
    zone's colour and labelled by its origin (ORIGIN_LABEL, "from A"), then
    "none" (in no zone at the reference). Only zones whose beta range
    overlaps the box are listed: beta does not change between stages, so
    the others would always count 0.
    Each zone is drawn with at most N_ZONE_DRAW random hits, the same hits
    at every stage; the drawn hits of all zones form one layer in a fixed
    random order, so no zone covers another. Each panel gives, per zone, the fraction of its hits
    with z > 0 (over all of the zone's hits, not only the drawn ones). A
    hit inside several boxes belongs to the first one in ZONES.
  * The (z, beta) population density of all stages in one figure: a 2D
    histogram per stage over the same N_SCATTER hits as the scatters,
    DENSITY_NBINS x DENSITY_NBINS bins, z in [-lim, lim] with lim = max |z|
    over all stages and these hits (or --zlim), beta in [0, 1]. Shared bins, axes
    and colour scale (YlGnBu, log counts unless --linear-counts), one colour
    bar; empty bins stay blank.
  * The same density with one sign of z per figure: z < 0 in
    z_vs_beta_density_negative, z > 0 in z_vs_beta_density_positive. Each
    side has DENSITY_NBINS z bins over its own range, [-lim_-, 0] and
    [0, lim_+] (lim = largest |z| of that sign over all stages and the shown
    hits, or --zlim), so twice the z resolution of the two-sided figure.
    Hits with z = 0 (g = 0) are in neither figure; each panel gives their
    number. Both figures share one colour scale, so their densities compare
    directly.
  * The same density figure with the raw gradient g instead of z: linear
    g bins over [-glim, glim], glim = max |g| over all stages and the same
    N_SCATTER hits (or --glim), same beta bins, colour map and colour scale rules. It shows
    what the transform hides: how strongly g piles up near zero, and how
    the gradient scale differs between stages.

Inputs: the stage CSVs in --probe-dir and the beta CSV of the same dump
(--beta-file; without it, beta.csv in --probe-dir or its parent, as in
gradient_statistics.py). Rows of the beta CSV must align with the stage CSVs.
Because all CSVs have the same rows, a hit is one row position at every
stage; zone tracking needs nothing else (no events.csv).

Outputs (in --out-dir/beta, --out-dir = the run directory)
----------------------------------------------------------
beta_histogram.{png,pdf}
z_vs_beta.{png,pdf}                         all stages, one scatter panel each
z_vs_beta/<i>_<stage>.{png,pdf}             one figure per stage, i = network order
z_vs_beta_zones.pdf                         all stages, zone hits coloured
z_ref_vs_z_stages.pdf                       z at REF_STAGE vs z at every other stage
z_vs_beta_zones_sign.csv                    per zone and stage: hit counts by sign of z,
                                            fraction still in the zone's z range,
                                            16/50/84th percentile of z
z_vs_beta_zones_boxes.csv                   per stage and box: hits inside, by reference zone
                                            (the box counts of z_vs_beta_zones)
z_vs_beta_zones_hits.csv                    event_id, hit_id, zone of every zone hit
                                            (among the N_SCATTER shown hits)
z_vs_beta_density.pdf                       all stages, one 2D histogram panel each
z_vs_beta_density_counts.csv                the counts of every bin (stage, bin edges, count)
z_vs_beta_density_negative.pdf              the same for z < 0 only, own z bins
z_vs_beta_density_negative_counts.csv       its bin counts
z_vs_beta_density_positive.pdf              the same for z > 0 only, own z bins
z_vs_beta_density_positive_counts.csv       its bin counts
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
cyan.jo, Claude Opus5.5 Extra, 2026-11-23: one-sided density figures
z_vs_beta_density_negative / _positive (z < 0 and z > 0 separately, own z
range and DENSITY_NBINS bins per side, z = 0 excluded and counted, one
colour scale for both) and their bin counts. plot_density_grid takes
precomputed counts, a fixed colour maximum, per-stage beta, panel texts and
a title note; density_meta stores the x range.
cyan.jo, Claude Opus5.5 Extra, 2026-11-27: zone tracking (ZONES, REF_STAGE,
N_ZONE_DRAW): z_vs_beta_zones, z_ref_vs_z_stages, the zone sign table and
the zone hit list; zone settings in beta_meta.json.
cyan.jo, Claude Opus5.5 Extra, 2026-11-27: zone figures: grey background =
hits in no zone only (gray, BACKGROUND_ALPHA = 0.15, just below
ZONE_ALPHA); the zones' drawn hits are one shuffled layer instead of one
layer per zone, which let the last zone drawn hide the others.
cyan.jo, Claude Opus5.5 Extra, 2026-11-27: zone boxes on every panel; per
box, the hits inside it at that stage by reference zone, written next to
the box (ZONE_TEXT_POS, ZONE_TEXT_FONTSIZE) and saved as
z_vs_beta_zones_boxes.csv. Each count line in its zone's colour
(getColour(k)), "none" in NONE_TEXT_COLOUR; no header line. Every zone is
listed in every box, each line labelled by its origin (ORIGIN_LABEL).
Then zones that can never reach a box (no beta overlap, always 0) are
left out of its text and of the CSV again.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
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
# beta bins 0.02). The raw-gradient density figure and each side of the
# one-sided figures use the same number.
DENSITY_NBINS = 50
# Colour map of the density figure.
DENSITY_CMAP = "YlGnBu"
# The two one-sided density figures: file suffix, hits kept, title note.
SIDES = (
    ("negative", np.less, r"$z < 0$"),
    ("positive", np.greater, r"$z > 0$"),
)

# Zone tracking. Boxes in the (z, beta) plane of REF_STAGE:
# name -> ((z_low, z_high), (beta_low, beta_high)), bounds inclusive.
# Zone i is drawn in getColour(i); a hit inside several boxes belongs to the
# first one listed.
ZONES = {
    "A": ((-6.0, -2.5), (0.8, 1.0)), # negative z, high beta
    # "B": ((-7.0, -1.0), (0.0, 0.2)), # negative z, low beta
    "B": ((-5.5, -4.5), (0.4, 0.8)), # negative z, low beta, post output tail
    "C": ((2.5, 6.0), (0.8, 1.0)),   # positive z, high beta
    # "D": ((1.0, 7.0), (0.0, 0.2)),   # positive z, low beta
    "D": ((4.0, 5.0), (0.0, 0.2)),   # positive z, low beta, post output tail
}
# Stage on which the zones are defined; None = the first stage shown.
REF_STAGE = "gradient_post_output"
# At most this many random hits per zone are drawn (the same at every stage);
# counts and fractions use all of the zone's hits.
N_ZONE_DRAW = None #20_000
ZONE_MARKER_SIZE = 2
ZONE_ALPHA = 0.1
# Box counts in z_vs_beta_zones: top-left corner of each box's text, in axes
# fractions (zones not listed get no text), its font size, the colour of the
# "none" line, and the label of each count line ({zone} = reference zone,
# {stage} = reference stage label, e.g. "{zone}@{stage}"). A block lists only
# the zones that can reach the box, plus "none"; C and D at x = 0.80 keep the
# lines inside the panel.
ZONE_TEXT_POS = {
    "A": (0.02, 0.78),
    "B": (0.02, 0.33),
    "C": (0.77, 0.78),
    "D": (0.77, 0.33),
}
ORIGIN_LABEL = "from {zone}"
ZONE_TEXT_FONTSIZE = 12
NONE_TEXT_COLOUR = "dimgray"
# Background of the zone figures: the shown hits in no zone (at REF_STAGE),
# a little fainter than the zone hits.
BACKGROUND_COLOUR = "gray"
BACKGROUND_ALPHA = 0.05


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


# -----------------------------------------------------------------------------
# Zone tracking
# -----------------------------------------------------------------------------
def check_zones(zones):
    """Every zone needs z_low < z_high and beta_low < beta_high."""
    for name, ((z0, z1), (b0, b1)) in zones.items():
        if not (z0 < z1 and b0 < b1):
            raise ValueError(f"Zone {name}: need z_low < z_high and beta_low < beta_high, "
                             f"got z [{z0}, {z1}], beta [{b0}, {b1}].")


def assign_zones(z_ref, beta, zones):
    """Zone of every hit on the reference stage.

    Returns (zone_of, n_multiple): zone_of[i] = position of the hit's zone in
    `zones`, -1 outside every box; bounds are inclusive. A hit inside several
    boxes belongs to the first one; n_multiple counts such hits.
    """
    zone_of = np.full(len(z_ref), -1, dtype=np.int64)
    n_boxes = np.zeros(len(z_ref), dtype=np.int64)
    for k, ((z0, z1), (b0, b1)) in enumerate(zones.values()):
        inside = (z_ref >= z0) & (z_ref <= z1) & (beta >= b0) & (beta <= b1)
        zone_of[inside & (zone_of < 0)] = k
        n_boxes += inside
    return zone_of, int((n_boxes > 1).sum())


def zone_draw_positions(zone_of, n_zones, n_draw, seed):
    """Per zone, the hit positions drawn in the zone figures: all of the
    zone's hits (n_draw None), or at most n_draw random ones. The same hits
    at every stage."""
    rng = np.random.default_rng(seed)
    drawn = []
    for k in range(n_zones):
        positions = np.flatnonzero(zone_of == k)
        if n_draw is not None and len(positions) > n_draw:
            positions = np.sort(rng.choice(positions, size=n_draw, replace=False))
        drawn.append(positions)
    return drawn


def zone_layer(drawn, seed):
    """The drawn hits of all zones as one layer: (positions, RGBA colours) in
    a fixed random order, the same in every panel. Drawn in one call, no
    zone is systematically on top of another."""
    positions = np.concatenate(drawn) if drawn else np.empty(0, dtype=np.int64)
    zone_index = np.concatenate([np.full(len(p), k) for k, p in enumerate(drawn)])
    palette = np.array([to_rgba(getColour(k), ZONE_ALPHA) for k in range(len(drawn))])
    order = np.random.default_rng(seed).permutation(len(positions))
    return positions[order], palette[zone_index[order]]


def zone_sign_text(z, zone_of, zones) -> dict:
    """Panel text: per zone, the fraction of its hits with z > 0 at this stage."""
    text = {}
    for k, name in enumerate(zones):
        z_zone = z[zone_of == k]
        text[f"{name}: z>0"] = f"{100 * (z_zone > 0).mean():.1f}%" if z_zone.size else "-"
    return text


def ranges_overlap(a, b) -> bool:
    """Closed intervals a = (low, high) and b share at least one point."""
    return a[0] <= b[1] and b[0] <= a[1]


def box_counts(z, beta, zone_of, zones) -> dict:
    """Hits inside every zone box at one stage, by their reference zone.

    Returns {box: {origin: count}}; origins are the zones whose beta range
    overlaps the box's, in ZONES order (beta does not change between
    stages, so any other zone would always count 0), then "none" = hits in
    no zone at the reference. Counts use all shown hits, not only the drawn
    ones.
    """
    counts = {}
    for box, ((z0, z1), (b0, b1)) in zones.items():
        inside = (z >= z0) & (z <= z1) & (beta >= b0) & (beta <= b1)
        by_origin = {origin: int((inside & (zone_of == k)).sum())
                     for k, (origin, (_, beta_range)) in enumerate(zones.items())
                     if ranges_overlap(beta_range, (b0, b1))}
        by_origin["none"] = int((inside & (zone_of < 0)).sum())
        counts[box] = by_origin
    return counts


def draw_box_counts(ax, x, y, by_origin, zones, ref_label):
    """The counts of one box at (x, y), axes fractions: one line per
    reference zone, labelled by ORIGIN_LABEL and drawn in that zone's colour
    (getColour of its position in ZONES), then "none" in NONE_TEXT_COLOUR.
    add_text_to_ax takes one colour per call, so every line is its own call,
    pushed down by leading newlines; all calls share the anchor and font, so
    the lines stay aligned.
    """
    names = list(zones)
    entries = [(ORIGIN_LABEL.format(zone=origin, stage=ref_label), n,
                getColour(names.index(origin)))
               for origin, n in by_origin.items() if origin != "none"]
    entries.append(("none", by_origin["none"], NONE_TEXT_COLOUR))
    width = max(len(label) for label, _, _ in entries) + 1
    lines = [(f"{label:<{width}}{n:>7,}", colour) for label, n, colour in entries]
    for i, (text, colour) in enumerate(lines):
        add_text_to_ax(x, y, "\n" * i + text, ax=ax,
                       fontsize=ZONE_TEXT_FONTSIZE, color=colour)


def box_count_table(z_by_stage, beta, zone_of, zones, labels) -> pd.DataFrame:
    """Long table of the box counts: stage, box, reference zone, hits."""
    rows = []
    for stage, z in z_by_stage.items():
        for box, by_origin in box_counts(z, beta, zone_of, zones).items():
            for origin, n in by_origin.items():
                rows.append({"stage": stage, "label": labels[stage], "box": box,
                             "origin": origin, "count": n})
    return pd.DataFrame(rows)


def zone_legend_handles(zones, n_in_zone, drawn):
    """Legend entries: the grey background, then every zone with its box and counts."""
    handles = [Line2D([], [], linestyle="none", marker="o", markersize=7,
                      color=BACKGROUND_COLOUR, label="hits in no zone")]
    for k, (name, ((z0, z1), (b0, b1))) in enumerate(zones.items()):
        handles.append(Line2D(
            [], [], linestyle="none", marker="o", markersize=7, color=getColour(k),
            label=(rf"{name}: $z \in [{z0:g}, {z1:g}]$, $\beta \in [{b0:g}, {b1:g}]$"
                   f"\n     {n_in_zone[k]:,} hits, {len(drawn[k]):,} drawn")))
    return handles


def finish_zone_grid(fig, axes, n_used, handles, legend_title, xlabel):
    """Legend in the first spare panel (or above the panels if none is spare),
    remove the other spare panels, one x label for the figure."""
    spare = list(axes.ravel()[n_used:])
    for ax in spare[1:]:
        fig.delaxes(ax)
    if spare:
        spare[0].axis("off")
        spare[0].legend(handles=handles, loc="center", title=legend_title,
                        frameon=False, fontsize=12, title_fontsize=13)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
    else:
        fig.legend(handles=handles, loc="upper center", ncol=len(handles),
                   title=legend_title, frameon=False, bbox_to_anchor=(0.5, 1.0))
        fig.tight_layout(rect=(0, 0.05, 1, 0.9))
    fig.supxlabel(xlabel, y=0.01, fontsize=plt.rcParams["axes.labelsize"])


def plot_z_vs_beta_zones(
    z_by_stage,
    beta,
    *,
    labels,
    zones,
    ref_stage,
    zone_of,
    drawn,
    zlim,
    log_scale=1e-8,
    beta_threshold=None,
    marker_size=1,
    seed=0,
    ncols=3,
):
    """(z, beta) of every stage, the hits of each reference-stage zone in
    their zone's colour over a grey background of the hits in no zone. The zone
    boxes are drawn on the reference panel. beta is a hit's own value at
    every stage, so a zone can only move along z. Returns (fig, axes).
    """
    stages = list(z_by_stage)
    nrows = int(np.ceil(len(stages) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    outside = zone_of < 0
    positions, colours = zone_layer(drawn, seed)
    for ax, stage in zip(axes.ravel(), stages):
        z = z_by_stage[stage]
        draw_scatter(ax, z[outside], beta[outside], colour=BACKGROUND_COLOUR, zlim=zlim,
                     beta_threshold=beta_threshold, marker_size=marker_size,
                     alpha=BACKGROUND_ALPHA)
        ax.scatter(z[positions], beta[positions], s=ZONE_MARKER_SIZE, c=colours,
                   linewidths=0, rasterized=True)
        # Boxes on every panel: dashed on the reference, dotted elsewhere.
        is_ref = stage == ref_stage
        for k, ((z0, z1), (b0, b1)) in enumerate(zones.values()):
            ax.add_patch(Rectangle((z0, b0), z1 - z0, b1 - b0, fill=False,
                                   edgecolor=getColour(k),
                                   linewidth=1.8 if is_ref else 1.2,
                                   linestyle="--" if is_ref else ":", zorder=5))
        # Hits inside each box at this stage, by reference zone.
        for box, by_origin in box_counts(z, beta, zone_of, zones).items():
            if box in ZONE_TEXT_POS:
                x, y = ZONE_TEXT_POS[box]
                draw_box_counts(ax, x, y, by_origin, zones, labels[ref_stage])
        ax.set_title(f"{labels[stage]} (reference)" if is_ref else labels[stage])
        # Top centre (z near 0, high beta) is empty in the real panels; the
        # top-left corner would cover zone A.
        add_text_to_ax(0.40, 0.97, nice_string_output(zone_sign_text(z, zone_of, zones)),
                       ax=ax, fontsize=12)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\beta$ value")

    n_in_zone = [int((zone_of == k).sum()) for k in range(len(zones))]
    finish_zone_grid(fig, axes, len(stages), zone_legend_handles(zones, n_in_zone, drawn),
                     f"Zones at {labels[ref_stage]}", z_label(log_scale))
    return fig, axes


def plot_zref_vs_zstage(
    z_by_stage,
    *,
    labels,
    zones,
    ref_stage,
    zone_of,
    drawn,
    zlim,
    marker_size=1,
    seed=0,
    ncols=3,
):
    """z at the reference stage (x) against z at every other stage (y), zone
    hits coloured as in z_vs_beta_zones. Dashed: y = x (z unchanged); dotted:
    y = -x. Hits in the upper-left and lower-right quadrants changed the sign
    of g. Returns (fig, axes), or None if the reference is the only stage.
    """
    stages = [s for s in z_by_stage if s != ref_stage]
    if not stages:
        return None
    z_ref = z_by_stage[ref_stage]
    nrows = int(np.ceil(len(stages) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 6.5 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    outside = zone_of < 0
    positions, colours = zone_layer(drawn, seed)
    for ax, stage in zip(axes.ravel(), stages):
        z = z_by_stage[stage]
        ax.scatter(z_ref[outside], z[outside], s=marker_size, alpha=BACKGROUND_ALPHA,
                   color=BACKGROUND_COLOUR, linewidths=0, rasterized=True)
        ax.scatter(z_ref[positions], z[positions], s=ZONE_MARKER_SIZE, c=colours,
                   linewidths=0, rasterized=True)
        ax.plot([-zlim, zlim], [-zlim, zlim], color="gray", linestyle="--", linewidth=0.8)
        ax.plot([-zlim, zlim], [zlim, -zlim], color="gray", linestyle=":", linewidth=0.8)
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.axvline(0, color="gray", linewidth=0.8)
        ax.set_xlim(-zlim, zlim)
        ax.set_ylim(-zlim, zlim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(labels[stage])
        add_text_to_ax(0.02, 0.97, nice_string_output(zone_sign_text(z, zone_of, zones)),
                       ax=ax, fontsize=12)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$z$ at the panel's stage")

    n_in_zone = [int((zone_of == k).sum()) for k in range(len(zones))]
    finish_zone_grid(fig, axes, len(stages), zone_legend_handles(zones, n_in_zone, drawn),
                     f"Zones at {labels[ref_stage]}",
                     rf"$z$ at {labels[ref_stage]} (reference)")
    return fig, axes


def zone_sign_table(z_by_stage, zone_of, zones, labels, ref_stage) -> pd.DataFrame:
    """Per zone and stage, over all of the zone's hits: counts by sign of z,
    the fraction still inside the zone's z range (beta does not change), and
    the 16th, 50th and 84th percentile of z."""
    rows = []
    for k, (name, ((z0, z1), _)) in enumerate(zones.items()):
        in_zone = zone_of == k
        for stage, z in z_by_stage.items():
            z_zone = z[in_zone]
            n = int(z_zone.size)
            q16, q50, q84 = (np.percentile(z_zone, [16, 50, 84]) if n
                             else (np.nan, np.nan, np.nan))
            rows.append({
                "zone": name, "stage": stage, "label": labels[stage],
                "reference": stage == ref_stage, "n": n,
                "n_pos": int((z_zone > 0).sum()), "n_neg": int((z_zone < 0).sum()),
                "n_zero": int((z_zone == 0).sum()),
                "frac_pos": (z_zone > 0).mean() if n else np.nan,
                "frac_neg": (z_zone < 0).mean() if n else np.nan,
                "frac_in_zone_z": ((z_zone >= z0) & (z_zone <= z1)).mean() if n else np.nan,
                "z_q16": q16, "z_median": q50, "z_q84": q84,
            })
    return pd.DataFrame(rows)


def symmetric_edges(values_by_stage, *, nbins=DENSITY_NBINS, lim=None):
    """nbins equal bins over [-lim, lim], shared by every stage.

    lim = the given lim, else the largest |value| over all stages, so the
    most extreme hit sits in an edge bin.
    """
    if lim is None:
        lim = max(float(np.abs(v).max()) for v in values_by_stage.values()) or 1.0
    return np.linspace(-lim, lim, nbins + 1)


def one_sided_edges(values_by_stage, side, *, nbins=DENSITY_NBINS, lim=None):
    """nbins equal bins over [-lim, 0] (side "negative") or [0, lim]
    ("positive"), shared by every stage.

    values_by_stage holds only the values of that side. lim = the given lim,
    else their largest |value| over all stages (1 if every stage is empty).
    """
    if lim is None:
        lim = max((float(np.abs(v).max()) for v in values_by_stage.values() if v.size),
                  default=0.0) or 1.0
    if side == "negative":
        return np.linspace(-lim, 0.0, nbins + 1)
    return np.linspace(0.0, lim, nbins + 1)


def beta_bin_edges(nbins=DENSITY_NBINS):
    """nbins equal beta bins over [0, 1]."""
    return np.linspace(0.0, 1.0, nbins + 1)


def density_counts(x_by_stage, beta, x_edges, beta_edges) -> dict:
    """2D histogram counts per stage, indexed [x bin, beta bin].

    beta is one array for every stage, or {stage: array} when each stage
    keeps different hits (the one-sided figures).
    """
    def beta_of(stage):
        return beta[stage] if isinstance(beta, dict) else beta

    return {stage: np.histogram2d(np.asarray(x, dtype=np.float64),
                                  np.asarray(beta_of(stage), dtype=np.float64),
                                  bins=(x_edges, beta_edges))[0]
            for stage, x in x_by_stage.items()}


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
    counts=None,
    vmax=None,
    panel_text=None,
    title_note=None,
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
    beta: one array, or {stage: array} when each stage keeps different hits.
    scientific_x writes the x ticks as multiples of a power of ten (raw g).
    extra_text: optional {stage: {key: text}} added to each panel's counts.
    counts: precomputed density_counts (default: computed here).
    vmax: largest count of the colour scale (default: largest bin here), so
    several figures can share one colour scale.
    panel_text: optional {stage: {key: text}} replacing the default hit
    counts (N, N+, N-) of a panel.
    title_note: optional text added to the figure title (e.g. "$z < 0$").
    Returns (fig, axes, counts); counts[stage] has shape
    (len(x_edges) - 1, len(beta_edges) - 1), indexed [x bin, beta bin].
    """
    if counts is None:
        counts = density_counts(x_by_stage, beta, x_edges, beta_edges)
    if vmax is None:
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
        if panel_text and stage in panel_text:
            text = dict(panel_text[stage])
        else:
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
    note = f", {title_note}" if title_note else ""
    fig.suptitle(
        rf"Population density of $({x_symbol}, \beta)${note}: {nx} $\times$ {nb} bins, "
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
    """Settings and result size of one density figure, for beta_meta.json.

    lim = the largest |edge|; x_range gives both ends (one-sided figures
    have one end at 0).
    """
    return {"nbins": len(x_edges) - 1, "n_points": int(n_points),
            "lim": float(max(abs(x_edges[0]), abs(x_edges[-1]))),
            "x_range": [float(x_edges[0]), float(x_edges[-1])],
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

    # Zone settings are checked before the (slow) reading.
    check_zones(ZONES)
    ref_stage = REF_STAGE or next(iter(stage_files))
    if ref_stage not in stage_files:
        raise SystemExit(
            f"REF_STAGE {ref_stage!r} is not among the stages read "
            f"({', '.join(stage_files)}); change REF_STAGE or --stages.")

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

    # 2b. Zone tracking over the same hits. All CSVs share their rows, so a
    #     hit is the same position in every stage's z array.
    beta_shown = beta[rows]
    zone_of, n_multiple = assign_zones(z_by_stage[ref_stage], beta_shown, ZONES)
    drawn = zone_draw_positions(zone_of, len(ZONES), N_ZONE_DRAW, args.seed)
    n_in_zone = {name: int((zone_of == k).sum()) for k, name in enumerate(ZONES)}
    print(f"Zones at {ref_stage}: "
          + ", ".join(f"{name} {n:,}" for name, n in n_in_zone.items())
          + f" of {len(rows):,} shown hits; {int((zone_of < 0).sum()):,} in no zone.")
    if n_multiple:
        print(f"WARNING: {n_multiple:,} hits lie in more than one zone box; "
              "each is kept in the first such zone (ZONES order).")

    fig, _ = plot_z_vs_beta_zones(
        z_by_stage, beta_shown, labels=labels, zones=ZONES, ref_stage=ref_stage,
        zone_of=zone_of, drawn=drawn, zlim=zlim, log_scale=args.log_scale,
        beta_threshold=args.beta_threshold, marker_size=args.marker_size,
        seed=args.seed)
    save_figure(fig, out_dir, "z_vs_beta_zones")
    plt.close(fig)

    result = plot_zref_vs_zstage(
        z_by_stage, labels=labels, zones=ZONES, ref_stage=ref_stage,
        zone_of=zone_of, drawn=drawn, zlim=zlim, marker_size=args.marker_size,
        seed=args.seed)
    if result is not None:
        save_figure(result[0], out_dir, "z_ref_vs_z_stages")
        plt.close(result[0])

    sign_table = zone_sign_table(z_by_stage, zone_of, ZONES, labels, ref_stage)
    sign_table.to_csv(out_dir / "z_vs_beta_zones_sign.csv", index=False)
    print("\n" + sign_table[["zone", "stage", "n", "frac_pos", "frac_in_zone_z",
                             "z_median"]].to_string(index=False, float_format="%.3f"))

    box_table = box_count_table(z_by_stage, beta_shown, zone_of, ZONES, labels)
    box_table.to_csv(out_dir / "z_vs_beta_zones_boxes.csv", index=False)
    print(f"Saved {out_dir / 'z_vs_beta_zones_boxes.csv'}")

    zone_names = np.array(list(ZONES))
    in_any = zone_of >= 0
    ref_frame = stage_frames[ref_stage]
    pd.DataFrame({
        "event_id": ref_frame["event_id"].to_numpy()[rows][in_any],
        "hit_id": ref_frame["hit_id"].to_numpy()[rows][in_any],
        "zone": zone_names[zone_of[in_any]],
    }).to_csv(out_dir / "z_vs_beta_zones_hits.csv", index=False)
    print(f"Saved {out_dir / 'z_vs_beta_zones_sign.csv'}")
    print(f"Saved {out_dir / 'z_vs_beta_zones_hits.csv'}")
    zone_info = {
        "ref_stage": ref_stage,
        "zones": {name: {"z": list(z_range), "beta": list(beta_range)}
                  for name, (z_range, beta_range) in ZONES.items()},
        "bounds": "inclusive; a hit in several boxes belongs to the first zone",
        "n_in_zone": n_in_zone,
        "n_no_zone": int((zone_of < 0).sum()),
        "n_in_several_boxes": n_multiple,
        "n_draw_max": N_ZONE_DRAW,
        "n_drawn": {name: int(len(drawn[k])) for k, name in enumerate(ZONES)},
        "colours": {name: f"getColour({k})" for k, name in enumerate(ZONES)},
        "background": "hits in no zone",
        "text_positions": {name: list(pos) for name, pos in ZONE_TEXT_POS.items()},
    }

    # 3. Population density over the scatter's N_SCATTER hits (the same rows
    #    at every stage and in both figures), DENSITY_NBINS bins per axis,
    #    shared edges and colour scale: (z, beta), then the raw gradient (g, beta).
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

    # One sign of z per figure, the same hits. Each side has DENSITY_NBINS z
    # bins over its own range; z = 0 (g = 0) is in neither side and counted
    # per panel. Counts of both sides first, so both share one colour scale.
    sides = {}
    for side, keep, note in SIDES:
        masks = {s: keep(z, 0.0) for s, z in z_by_stage.items()}
        z_side = {s: z_by_stage[s][m] for s, m in masks.items()}
        beta_side = {s: beta_shown[m] for s, m in masks.items()}
        edges = one_sided_edges(z_side, side, nbins=DENSITY_NBINS, lim=args.zlim)
        sides[side] = {
            "note": note, "z": z_side, "beta": beta_side, "edges": edges,
            "counts": density_counts(z_side, beta_side, edges, beta_edges),
            "text": {s: {"N": f"{len(z_side[s]):,}",
                         "of": f"{len(rows):,}",
                        #  "z = 0": f"{int((z_by_stage[s] == 0).sum()):,}"
                         }
                     for s in z_by_stage},
        }
    vmax_sides = max(1.0, max(float(c.max()) for d in sides.values()
                              for c in d["counts"].values()))
    for side, d in sides.items():
        edges = d["edges"]
        print(f"Density (z, {side}): z in [{edges[0]:.3f}, {edges[-1]:.3f}] "
              f"(bin width {edges[1] - edges[0]:.3f}), hits per stage "
              + ", ".join(f"{s} {len(z):,}" for s, z in d["z"].items()) + ".")
        fig, _, _ = plot_density_grid(
            d["z"], d["beta"], x_edges=edges, xlabel=z_label(args.log_scale),
            x_symbol="z", counts=d["counts"], vmax=vmax_sides,
            panel_text=d["text"], title_note=d["note"], **density_kwargs)
        save_figure(fig, out_dir, f"z_vs_beta_density_{side}")
        plt.close(fig)
        density_counts_table(d["counts"], labels, edges, beta_edges, x_name="z").to_csv(
            out_dir / f"z_vs_beta_density_{side}_counts.csv", index=False)
        density_info[f"z_{side}"] = {
            **density_meta(d["counts"], edges, beta_edges, n_points=len(rows),
                           linear_counts=args.linear_counts),
            "n_on_side": {s: int(len(z)) for s, z in d["z"].items()},
            "n_zero": {s: int((z_by_stage[s] == 0).sum()) for s in z_by_stage},
            "colour_max_shared": vmax_sides,
        }
    del sides

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
        "zones": zone_info,
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
                        help="z range of the scatter and density figures: [-zlim, zlim], "
                             "and [-zlim, 0] / [0, zlim] in the one-sided density figures "
                             "(default: scatter 1.05 x max |z|, density max |z|, one-sided "
                             "max |z| of that sign, all of the shown hits)")
    parser.add_argument("--glim", type=float, default=None,
                        help="g range [-glim, glim] of the raw-gradient density figure "
                             "(default: max |g| of the shown hits over all stages)")
    parser.add_argument("--linear-counts", action="store_true",
                        help="linear colour scale in the density figures (default: log)")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed of the N_SCATTER subsample of the scatter and density "
                             "figures and of the drawn zone hits (default 0)")
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