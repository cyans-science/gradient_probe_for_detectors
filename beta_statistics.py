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

Inputs: the stage CSVs in --probe-dir and the beta CSV of the same dump
(--beta-file; without it, beta.csv in --probe-dir or its parent, as in
gradient_statistics.py). Rows of the beta CSV must align with the stage CSVs.

Outputs (in --out-dir/beta, --out-dir = the run directory)
----------------------------------------------------------
beta_histogram.{png,pdf}
z_vs_beta.{png,pdf}                         all stages, one scatter panel each
z_vs_beta/<i>_<stage>.{png,pdf}             one figure per stage, i = network order
beta_summary.csv                            per stage: hits drawn, N+, N-
beta_meta.json

cyan.jo, Claude Opus5.5 Extra, 2026-11-19
cyan.jo, Claude Opus5.5 Extra, 2026-11-19: --beta-file; bin number fixed in
the script (BETA_NBINS) instead of --bins; one (z, beta) figure per stage
instead of one grid figure; no correlation coefficient (distribution only);
marginal histograms of beta (left) and z (bottom) around each scatter;
always N_SCATTER = 1,000,000 random hits in the scatter figures (no N0);
all-stage figure z_vs_beta (scatter panels only) next to the per-stage ones.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
        draw_scatter(ax, z_by_stage[stage], beta, colour=getColour(i % 10), zlim=zlim,
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
                        help="x range of the scatter is [-zlim, zlim] (default: 1.05 x max |z|)")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed of the N_SCATTER subsample of the (z, beta) figures (default 0)")
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