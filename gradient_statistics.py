import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append('Utils/')
from PlotUtils import setMplParam, getColour, getHistoParam, annotate_histogram, enable_histogram_counts
from ExternalFunctions import add_text_to_ax, nice_string_output
import matplotlib.pyplot as plt
    
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
            event_id : int
            hit_id   : int
            value    : float
    """
    probe_file = Path(probe_file)

    if not probe_file.exists():
        raise FileNotFoundError(f"Probe file not found: {probe_file}")

    df = pd.read_csv(probe_file)

    expected_columns = {"event_id", "hit_id", "value"}
    missing_columns = expected_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"{probe_file.name} is missing columns: {sorted(missing_columns)}"
        )

    df = df[["event_id", "hit_id", "value"]].copy()

    df["event_id"] = df["event_id"].astype(int)
    df["hit_id"] = df["hit_id"].astype(int)
    df["value"] = df["value"].astype(float)

    return df

def get_highest_values(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Top-n rows among the per-event maxima, ranked by |value| (sign kept)."""
    score = df["value"].abs()
    idx = score.groupby(df["event_id"]).idxmax()          # per-event max |value|
    return df.loc[score.loc[idx].nlargest(n).index]       # top-n events, signed rows

def track_top_hits(
    df_dict: dict[str, pd.DataFrame], n: int = 20, min_stages: int = 1
) -> pd.DataFrame:
    """
    Follow the hits that are in the top-n (by |value|) at any stage across all stages.

    Parameters
    ----------
    df_dict : dict[str, pd.DataFrame]
        Stage name -> DataFrame (event_id, hit_id, value), in chronological order.
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
    keys_per_stage = {
        name: get_highest_values(df, n)[["event_id", "hit_id"]]
        for name, df in df_dict.items()
    }
    candidates = pd.MultiIndex.from_frame(
        pd.concat(keys_per_stage.values()).drop_duplicates()
    )

    wide = pd.DataFrame({
        name: df.set_index(["event_id", "hit_id"])["value"].reindex(candidates)
        for name, df in df_dict.items()
    })
    wide.columns.name = "stage"

    stages = list(df_dict)
    top_sets = {s: set(map(tuple, keys_per_stage[s].values)) for s in stages}
    wide["peak_stage"] = wide[stages].abs().idxmax(axis=1)
    wide["top_in"] = [[s for s in stages if key in top_sets[s]] for key in wide.index]
    wide["n_top"] = wide["top_in"].str.len()

    wide = wide[wide["n_top"] >= min_stages]
    return wide.sort_values(stages[-1], ascending=False, key=lambda s: s.abs())

def plot_top_hits_evolution(
    stages: dict[str, pd.DataFrame], n: int = 20, min_stages: int = 2
):
    from matplotlib.ticker import FuncFormatter
    from matplotlib.lines import Line2D

    wide = track_top_hits(stages, n=n, min_stages=min_stages)
    stage_cols = list(stages)
    if wide.empty:
        print(f"No hit is in the top-{n} of >= {min_stages} stages.")
        return wide

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
    ax.set_xticklabels(stage_cols)
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
    plt.show()
    return wide

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
    values = np.asarray(values)
    return np.sign(values) * np.log10(1 + np.abs(values) / scale)

def inverse_signed_log10(z, scale=1e-6):
    z = np.asarray(z)
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

def run(source_dir:Path, save_dir:Path):
    probe_files = {
        "initial": next(source_dir.glob("*initial*.csv")),
        "post_gn1": next(source_dir.glob("*post_gn1*.csv")),
        "post_gn2": next(source_dir.glob("*post_gn2*.csv")),
        "post_gn3": next(source_dir.glob("*post_gn3*.csv")),
        "post_gn4": next(source_dir.glob("*post_gn4*.csv")),
        "output": next(source_dir.glob("*output*.csv")),
    }
    df_initial = read_probe_file(probe_files["initial"])
    df_post_gn1 = read_probe_file(probe_files["post_gn1"])
    df_post_gn2 = read_probe_file(probe_files["post_gn2"])
    df_post_gn3 = read_probe_file(probe_files["post_gn3"])
    df_post_gn4 = read_probe_file(probe_files["post_gn4"])
    df_output = read_probe_file(probe_files["output"])
    
    probe_gradients = {
        "@Initial Processing": df_initial,
        "@After GN1": df_post_gn1,
        "@After GN2": df_post_gn2,
        "@After GN3": df_post_gn3,
        "@After GN4": df_post_gn4,
        "@After Output Head": df_output,
    }

    fig, ax = plot_gradient_histograms_overlay(
        probe_gradients,
        title="Gradient distribution across probing locations",
        log_scale=1e-8,
        Nbins=100,
        save_dir=save_dir
        )

def extract_run_date(path) -> str:
    """
    Return the first YYYYMMDD date found in `path`,
    e.g. '20261019' from '.../20261019_Fisher_depth_probe/...'.
    """
    import re
    from datetime import datetime

    for candidate in re.findall(r"(?<!\d)\d{8}(?!\d)", str(path)):
        try:
            datetime.strptime(candidate, "%Y%m%d")
            return candidate
        except ValueError:
            continue

    raise ValueError(f"No YYYYMMDD date found in path: {path}")

def main():
    # plot style settings
    setMplParam()
    enable_histogram_counts()
    
    source_dir = Path(
        "/work/cjo/e2ereco_topas/20261019_Fisher_depth_probe/Results/20261019_Fisher_depth_probe_per_hit/grad"
        )
    run_date = extract_run_date(source_dir)
    save_dir = Path(
        f"/ceph/cjo/GradientProbe/{run_date}"
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    run(source_dir=source_dir,save_dir=save_dir)
    
    
if __name__ == "__main__":
    main()