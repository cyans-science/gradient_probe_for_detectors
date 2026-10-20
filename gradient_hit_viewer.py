
import sys
import os
import io
import shutil
import subprocess

from pathlib import Path
import numpy as np
import pandas as pd

import uproot
from particle import Particle

sys.path.append('Utils/')
from PlotUtils import setMplParam, getColour, getHistoParam, annotate_histogram
from ExternalFunctions import add_text_to_ax, nice_string_output

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation, HTMLWriter, FFMpegWriter
from PIL import Image
import importlib.util

def load_converted_events(root_file, converter, max_events=None):
    converter = converter

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

    for event_idx in range(len(hits)):

        if len(truth["pid"][event_idx]) == 0:
            print(
                f"Event {event_idx}: "
                "no SimShowers -> 0 converted rows"
            )

            event_df = pd.DataFrame()

        else:
            event_df = converter.convertevent(
                hits[event_idx],
                showers[event_idx],
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

DETECTOR_COLORS = {
    "PbWO4": "royalblue",
    "Brass": "darkgoldenrod",
    "Scintillator": "seagreen",
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

    for material, points in geom["lines"]:
        ax.plot(
            points[:, 0], points[:, 1], points[:, 2],
            color=DETECTOR_COLORS[material],
            alpha=wire_alpha,
            lw=0.6,
            label="_nolegend_",
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


def plot_event_hits_3d(
    df, row_splits, event_id,
    show_pid=False,
    detector=None,
    view="event",
    coordinate_unit=None,
    vertical_axis="z",
    elev=60, azim=100,
    zoom_region=None, 
    show=True
):
    if not 0 <= event_id < len(row_splits) - 1:
        raise IndexError(f"event_id must be between 0 and {len(row_splits) - 2}")

    start, stop = map(int, row_splits[event_id:event_id + 2])
    hits = df.iloc[start:stop]
    
    total_hits = len(hits)
    zoom_bounds = None

    if zoom_region is not None:
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

        positions = hits[["recHitX", "recHitY", "recHitZ"]].to_numpy()
        inside = (
            (positions >= zoom_bounds[0])
            & (positions <= zoom_bounds[1])
        ).all(axis=1)

        hits = hits.loc[inside]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    
    if view not in {"event", "detector"}:
        raise ValueError("view must be 'event' or 'detector'")

    if view == "detector" and detector is None:
        raise ValueError("view='detector' requires a detector configuration")

    # Capture hit-only limits before drawing the detector.
    # This keeps the event view centred on the shower.
    xyz = hits[["recHitX", "recHitY", "recHitZ"]].to_numpy(dtype=float)
    xyz = xyz[np.isfinite(xyz).all(axis=1)]

    event_bounds = (
        np.stack([xyz.min(axis=0), xyz.max(axis=0)])
        if len(xyz) else None
    )

    detector_bounds = (
        draw_detector_3d(ax, **detector)
        if detector is not None else None
    )
    if zoom_bounds is not None and detector is not None:
        clip_detector_lines_to_box(ax, zoom_bounds)

    pid_markers = {
        22: "*",                  # photon
        11: "^",  -11: "^",       # electron / positron
        13: "v",  -13: "v",       # muon / antimuon
        211: "o", -211: "o",      # charged pion
        321: "D", -321: "D",      # charged kaon
        2212: "s", -2212: "s",    # proton / antiproton
        2112: "p", -2112: "p",    # neutron / antineutron
    }

    def marker_size(energy):
        return np.clip(8 + 12 * np.log10(1 + np.maximum(energy, 0)), 8, 75)

    def particle_label(pid):
        try:
            name = Particle.from_pdgid(int(pid)).latex_name
        except (ValueError, LookupError, AttributeError):
            return f"PID {int(pid)}"
        return name if name.startswith("$") else f"${name}$"

    for object_id, object_hits in hits.groupby("t_idx"):
        color = (
            "lightgray" if object_id < 0
            else plt.get_cmap("tab20")(int(object_id) % 20)
        )

        for pid, group in object_hits.groupby("t_pid"):
            marker = "x" if object_id < 0 else pid_markers.get(int(pid), "X")
            label = (
                "unassigned" if object_id < 0
                else f"object {int(object_id)}: {particle_label(pid)}"
            )

            ax.scatter(
                group["recHitX"],
                group["recHitY"],
                group["recHitZ"],
                s=marker_size(group["recHitEnergy"]),
                marker=marker,
                color=color,
                alpha=0.75,
                label=label,
                depthshade=False,
            )

            if show_pid:
                for hit in group.itertuples(index=False):
                    ax.text(
                        hit.recHitX, hit.recHitY, hit.recHitZ,
                        str(int(hit.t_pid)),
                        fontsize=7,
                    )

    unit_label = f" [{coordinate_unit}]" if coordinate_unit else ""
    ax.set(
        xlabel=f"recHitX{unit_label}",
        ylabel=f"recHitY{unit_label}",
        zlabel=f"recHitZ{unit_label}",
        title=(
            f"Event {event_id}: {len(hits)} / {total_hits} hits in zoom region"
            if zoom_bounds is not None
            else f"Event {event_id}: {total_hits} hits"
        ),
    )
    ax.view_init(elev=elev, 
                 azim=azim,
                 vertical_axis=vertical_axis,)

    if zoom_bounds is not None:
        # Explicit zoom overrides automatic event/detector bounds.
        bounds = zoom_bounds.copy()

    elif view == "detector":
        bounds = detector_bounds.copy()
        if event_bounds is not None:
            bounds[0] = np.minimum(bounds[0], event_bounds[0])
            bounds[1] = np.maximum(bounds[1], event_bounds[1])

    else:
        bounds = event_bounds if event_bounds is not None else detector_bounds

    if bounds is not None:
        center = bounds.mean(axis=0)
        span = bounds[1] - bounds[0]

        if zoom_bounds is None:
            span = np.maximum(span, max(span.max() * 0.02, 1e-6))
            span *= 1.10

        ax.set_xlim(center[0] - span[0] / 2, center[0] + span[0] / 2)
        ax.set_ylim(center[1] - span[1] / 2, center[1] + span[1] / 2)
        ax.set_zlim(center[2] - span[2] / 2, center[2] + span[2] / 2)
        ax.set_box_aspect(span)

    if not hits.empty:
        # Keep your existing energy-size and object legend code here,
        # starting with:
        energies = hits["recHitEnergy"].clip(lower=0).to_numpy()

        # Show representative energies from this event in a size legend.
        energies = hits["recHitEnergy"].clip(lower=0).to_numpy()
        examples = np.unique(np.quantile(energies, [0, 0.5, 1]))

        size_handles = [
            ax.scatter(
                [], [], [],
                s=marker_size(energy),
                marker="o",
                facecolor="white",
                edgecolor="black",
                # label=f"{energy:.3g}",
                label="_nolegend_",
            )
            for energy in examples
        ]

                # Collect plot entries before adding either legend.
        handles, labels = ax.get_legend_handles_labels()

        # Keep one entry for repeated labels, especially "unassigned".
        unique_entries = dict(zip(labels, handles))
        unique_entries.pop("_nolegend_", None)

        object_legend = fig.legend(
            unique_entries.values(),
            unique_entries.keys(),
            title="True objects",
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

    ax.tick_params(axis="both", which="major", labelsize=8)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.label.set_fontsize(10)

    fig.subplots_adjust(left=0.02, right=0.77, bottom=0.05, top=0.95)
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


def render_rotation_frames(
    fig, ax,
    *,
    elev=10,
    vertical_axis="y",
    start_azim=0,
    n_frames=60,
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
    
def save_frames(
    frames,
    save_path,
    *,
    formats=("mp4",),
    fps=20,
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
                str(path),
            ]
            result = subprocess.run(
                command,
                input=b"".join(frame.tobytes() for frame in frames),
                capture_output=True,
            )
            if result.returncode != 0:
                print(
                    f"ffmpeg failed -> skipping mp4 "
                    f"(try `pip install imageio-ffmpeg`):\n"
                    f"{result.stderr.decode(errors='replace')}"
                )
                continue
 
        elif fmt in ("webp", "gif"):
            images, options = frames, {"quality": webp_quality, "method": 6}
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
                path,
                save_all=True,
                append_images=images[1:],
                duration=frame_ms,
                loop=0,
                **options,
            )
 
        else:
            raise ValueError(f"Unknown format {fmt!r}; use 'mp4', 'webp' or 'gif'.")
 
        saved.append(path)
        print(f"Saved animation: {path} ({path.stat().st_size / 1e6:.1f} MB)")
 
    return saved
    
def run_animation(
    df,
    row_splits,
    event_id,
    *,
    save_path,
    formats=("mp4",),
    dpi=100,
    crf=20,
    webp_quality=80,
    animation_kwargs=None,
    **plot_kwargs,
):
    """Render a rotating 3D event view and save it.
 
    save_path: output path without (or with any) suffix; the suffix is
        replaced per format, e.g. ".../event1_detector" -> event1_detector.mp4
    formats: any of "mp4", "webp", "gif" (see save_frames). The frames are
        rendered once, so extra formats only cost encoding time.
    animation_kwargs: elev, vertical_axis, start_azim, n_frames, fps.
    """
    fig, ax = plot_event_hits_3d(
        df, row_splits, event_id,
        show=False,
        **plot_kwargs,
    )
 
    # Keep the plot's camera settings unless explicitly overridden.
    rotation_kwargs = {
        "elev": plot_kwargs.get("elev", 60),
        "vertical_axis": plot_kwargs.get("vertical_axis", "z"),
        "start_azim": plot_kwargs.get("azim", 100),
    }
    rotation_kwargs.update(animation_kwargs or {})
    fps = rotation_kwargs.pop("fps", 20)
 
    frames = render_rotation_frames(fig, ax, dpi=dpi, **rotation_kwargs)
    plt.close(fig)
 
    return save_frames(
        frames, save_path,
        formats=formats,
        fps=fps,
        crf=crf,
        webp_quality=webp_quality,
    )


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


def save_event_hits_3d_html(
    df, row_splits, event_id,
    *,
    save_path,
    detector=None,
    view="event",
    coordinate_unit=None,
    vertical_axis="z",
    elev=60, azim=100,
    zoom_region=None,
    wire_opacity=0.35,
    hit_marker_scale=1.5,
    include_plotlyjs=True,
):
    """Save an interactive (rotate/zoom/pan/hover) plotly view of one event.

    Takes the same view arguments as plot_event_hits_3d. Click legend
    entries to hide/show individual objects or detector materials.
    include_plotlyjs: True embeds plotly.js (~4 MB, works offline);
        "cdn" keeps the file small but needs internet to open.
    """
    import plotly.graph_objects as go

    if not 0 <= event_id < len(row_splits) - 1:
        raise IndexError(f"event_id must be between 0 and {len(row_splits) - 2}")
    if view not in {"event", "detector"}:
        raise ValueError("view must be 'event' or 'detector'")
    if view == "detector" and detector is None:
        raise ValueError("view='detector' requires a detector configuration")

    start, stop = map(int, row_splits[event_id:event_id + 2])
    hits = df.iloc[start:stop]
    total_hits = len(hits)
    zoom_bounds = None

    if zoom_region is not None:
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

        positions = hits[["recHitX", "recHitY", "recHitZ"]].to_numpy()
        inside = (
            (positions >= zoom_bounds[0])
            & (positions <= zoom_bounds[1])
        ).all(axis=1)

        hits = hits.loc[inside]

    xyz = hits[["recHitX", "recHitY", "recHitZ"]].to_numpy(dtype=float)
    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    event_bounds = (
        np.stack([xyz.min(axis=0), xyz.max(axis=0)])
        if len(xyz) else None
    )

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

    def marker_size(energy):
        # Same area scale as the matplotlib plot; plotly sizes are diameters.
        area = np.clip(8 + 12 * np.log10(1 + np.maximum(energy, 0)), 8, 75)
        return hit_marker_scale * np.sqrt(area)

    def particle_name(pid):
        try:
            return Particle.from_pdgid(int(pid)).name
        except (ValueError, LookupError, AttributeError):
            return f"PID {int(pid)}"

    unit_label = f" [{coordinate_unit}]" if coordinate_unit else ""
    shown_labels = set()

    for object_id, object_hits in hits.groupby("t_idx"):
        color = (
            "lightgray" if object_id < 0
            else mpl.colors.to_hex(plt.get_cmap("tab20")(int(object_id) % 20))
        )

        for pid, group in object_hits.groupby("t_pid"):
            symbol = "x" if object_id < 0 else PLOTLY_PID_SYMBOLS.get(int(pid), "x")
            label = (
                "unassigned" if object_id < 0
                else f"object {int(object_id)}: {particle_name(pid)}"
            )
            energy = group["recHitEnergy"].to_numpy(dtype=float)

            traces.append(go.Scatter3d(
                x=group["recHitX"], y=group["recHitY"], z=group["recHitZ"],
                mode="markers",
                marker=dict(
                    size=marker_size(energy),
                    symbol=symbol,
                    color=color,
                    opacity=0.75,
                    line=dict(width=0),
                ),
                name=label,
                legendgroup=label,
                # "unassigned" spans several PIDs; show it once in the legend.
                showlegend=label not in shown_labels,
                customdata=np.column_stack([energy, group["t_pid"]]),
                hovertemplate=(
                    f"{label}<br>"
                    "recHitEnergy=%{customdata[0]:.3g}<br>"
                    "t_pid=%{customdata[1]:d}<br>"
                    f"x=%{{x:.1f}}, y=%{{y:.1f}}, z=%{{z:.1f}}{unit_label}"
                    "<extra></extra>"
                ),
            ))
            shown_labels.add(label)

    if zoom_bounds is not None:
        bounds = zoom_bounds.copy()
    elif view == "detector":
        bounds = detector_bounds.copy()
        if event_bounds is not None:
            bounds[0] = np.minimum(bounds[0], event_bounds[0])
            bounds[1] = np.maximum(bounds[1], event_bounds[1])
    else:
        bounds = event_bounds if event_bounds is not None else detector_bounds

    scene = dict(
        xaxis_title=f"recHitX{unit_label}",
        yaxis_title=f"recHitY{unit_label}",
        zaxis_title=f"recHitZ{unit_label}",
    )

    if bounds is not None:
        center = bounds.mean(axis=0)
        span = bounds[1] - bounds[0]

        if zoom_bounds is None:
            span = np.maximum(span, max(span.max() * 0.02, 1e-6))
            span *= 1.10

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
        title=(
            f"Event {event_id}: {len(hits)} / {total_hits} hits in zoom region"
            if zoom_bounds is not None
            else f"Event {event_id}: {total_hits} hits"
        ),
        scene=scene,
        template="plotly_white",
        legend=dict(itemsizing="constant", groupclick="toggleitem"),
        height=800,
        margin=dict(l=0, r=0, t=50, b=0),
    )

    html_path = Path(save_path).with_suffix(".html")
    html_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(html_path, include_plotlyjs=include_plotlyjs)
    print(
        f"Saved interactive view: {html_path} "
        f"({html_path.stat().st_size / 1e6:.1f} MB)"
    )
    return html_path

 
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
 
 
def save_event_hits_3d_html(
    df, row_splits, event_id,
    *,
    save_path,
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
 
    Takes the same view arguments as plot_event_hits_3d. Click legend
    entries to hide/show individual objects or detector materials.
    include_plotlyjs: True embeds plotly.js (~4 MB, works offline);
        "cdn" keeps the file small but needs internet to open.
    auto_rotate: start a turntable rotation when the page opens. A button
        toggles it; dragging the scene stops it.
    """
    import plotly.graph_objects as go
 
    if not 0 <= event_id < len(row_splits) - 1:
        raise IndexError(f"event_id must be between 0 and {len(row_splits) - 2}")
    if view not in {"event", "detector"}:
        raise ValueError("view must be 'event' or 'detector'")
    if view == "detector" and detector is None:
        raise ValueError("view='detector' requires a detector configuration")
 
    start, stop = map(int, row_splits[event_id:event_id + 2])
    hits = df.iloc[start:stop]
    total_hits = len(hits)
    zoom_bounds = None
 
    if zoom_region is not None:
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
 
        positions = hits[["recHitX", "recHitY", "recHitZ"]].to_numpy()
        inside = (
            (positions >= zoom_bounds[0])
            & (positions <= zoom_bounds[1])
        ).all(axis=1)
 
        hits = hits.loc[inside]
 
    xyz = hits[["recHitX", "recHitY", "recHitZ"]].to_numpy(dtype=float)
    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    event_bounds = (
        np.stack([xyz.min(axis=0), xyz.max(axis=0)])
        if len(xyz) else None
    )
 
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
 
    def marker_size(energy):
        # Same area scale as the matplotlib plot; plotly sizes are diameters.
        area = np.clip(8 + 12 * np.log10(1 + np.maximum(energy, 0)), 8, 75)
        return hit_marker_scale * np.sqrt(area)
 
    def particle_name(pid):
        try:
            return Particle.from_pdgid(int(pid)).name
        except (ValueError, LookupError, AttributeError):
            return f"PID {int(pid)}"
 
    unit_label = f" [{coordinate_unit}]" if coordinate_unit else ""
    shown_labels = set()
 
    for object_id, object_hits in hits.groupby("t_idx"):
        color = (
            "lightgray" if object_id < 0
            else mpl.colors.to_hex(plt.get_cmap("tab20")(int(object_id) % 20))
        )
 
        for pid, group in object_hits.groupby("t_pid"):
            symbol = "x" if object_id < 0 else PLOTLY_PID_SYMBOLS.get(int(pid), "x")
            label = (
                "unassigned" if object_id < 0
                else f"object {int(object_id)}: {particle_name(pid)}"
            )
            energy = group["recHitEnergy"].to_numpy(dtype=float)
 
            traces.append(go.Scatter3d(
                x=group["recHitX"], y=group["recHitY"], z=group["recHitZ"],
                mode="markers",
                marker=dict(
                    size=marker_size(energy),
                    symbol=symbol,
                    color=color,
                    opacity=0.75,
                    line=dict(width=0),
                ),
                name=label,
                legendgroup=label,
                # "unassigned" spans several PIDs; show it once in the legend.
                showlegend=label not in shown_labels,
                customdata=np.column_stack([energy, group["t_pid"]]),
                hovertemplate=(
                    f"{label}<br>"
                    "recHitEnergy=%{customdata[0]:.3g}<br>"
                    "t_pid=%{customdata[1]:d}<br>"
                    f"x=%{{x:.1f}}, y=%{{y:.1f}}, z=%{{z:.1f}}{unit_label}"
                    "<extra></extra>"
                ),
            ))
            shown_labels.add(label)
 
    if zoom_bounds is not None:
        bounds = zoom_bounds.copy()
    elif view == "detector":
        bounds = detector_bounds.copy()
        if event_bounds is not None:
            bounds[0] = np.minimum(bounds[0], event_bounds[0])
            bounds[1] = np.maximum(bounds[1], event_bounds[1])
    else:
        bounds = event_bounds if event_bounds is not None else detector_bounds
 
    scene = dict(
        xaxis_title=f"recHitX{unit_label}",
        yaxis_title=f"recHitY{unit_label}",
        zaxis_title=f"recHitZ{unit_label}",
    )
 
    if bounds is not None:
        center = bounds.mean(axis=0)
        span = bounds[1] - bounds[0]
 
        if zoom_bounds is None:
            span = np.maximum(span, max(span.max() * 0.02, 1e-6))
            span *= 1.10
 
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
        title=(
            f"Event {event_id}: {len(hits)} / {total_hits} hits in zoom region"
            if zoom_bounds is not None
            else f"Event {event_id}: {total_hits} hits"
        ),
        title_x=0.5,  # keeps the title clear of the rotation button
        scene=scene,
        template="plotly_white",
        legend=dict(itemsizing="constant", groupclick="toggleitem"),
        height=800,
        margin=dict(l=0, r=0, t=50, b=0),
    )
 
    rotation_js = (
        PLOTLY_ROTATION_JS
        .replace("__SPEED__", repr(float(rotation_deg_per_s)))
        .replace("__AUTOSTART__", "true" if auto_rotate else "false")
    )
 
    html_path = Path(save_path).with_suffix(".html")
    html_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        html_path,
        include_plotlyjs=include_plotlyjs,
        post_script=rotation_js,
    )
    print(
        f"Saved interactive view: {html_path} "
        f"({html_path.stat().st_size / 1e6:.1f} MB)"
    )
    return html_path
 
 
def run(source_dir:Path,save_dir:Path):
    DETECTOR = {
        "barrel_inner_radius_cm": 150.0,  # Provisional: verify physical radius.
        "barrel_half_length_cm": 300.0,
        "endcap_front_z_cm": 300.0,       # Positive |z| at the ECAL entrance.
        "endcap_inner_radius_cm": 30.0,
        "endcap_outer_radius_cm": 150.0,
        "coordinate_units_per_cm": 10,  # 1 for cm; 10 for mm.
    }
 
    path = (
        "/work/cjo/e2ereco_bernoulli_reproduction/source/"
        "modules/datastructures/TrainData_DICE_cellparent.py"
    )
 
    spec = importlib.util.spec_from_file_location(
        "TrainData_DICE_cellparent",
        path
    )
 
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
 
    base_dir_root="/ceph/cjo/DICEv1_4_0/SingleJet_pt30_eta1p2_1p7_100k/"
    root_file_name="singlejet_11.root"
    TrainData_DICE_cellparent = module.TrainData_DICE_cellparent
    df, row_splits, truth = load_converted_events(
        root_file=os.path.join(base_dir_root, root_file_name),
        converter=TrainData_DICE_cellparent(),
        max_events=100,
    )
    event_id = 1
    views = {
        "detector": dict(
            detector=DETECTOR,  # Enables the physical layer wireframes.
            view="detector",    # Fits the whole detector and hits in the axes.
            coordinate_unit="cm",
            elev=10,
            azim=0,
            vertical_axis="y",
        ),
        "zoom": dict(
            detector=DETECTOR,
            view="detector",
            coordinate_unit="cm",
            elev=5,
            azim=0,
            vertical_axis="y",
            zoom_region={
                "x": (1000, 2000),
                "y": (500, 1500),
                "z": (750, 1750),
            },
        ),
    }
 
    for name, view_kwargs in views.items():
        save_path = save_dir / f"event{event_id}_{name}"
 
        # Interactive plotly view with a built-in turntable rotation.
        save_event_hits_3d_html(
            df, row_splits, event_id,
            save_path=save_path.with_name(f"{save_path.name}_interactive"),
            **view_kwargs,
        )
        # Pre-rendered rotation: mp4 for viewing/slides, webp for
        # dropping inline into Markdown, notebooks or chat.
        run_animation(
            df, row_splits, event_id,
            save_path=save_path,
            formats=("mp4", "webp"),
            **view_kwargs,
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
    font_dir = Path(mpl.get_data_path()) / "fonts" / "ttf"
 
    for font_path in font_dir.glob("*.ttf"):
        fm.fontManager.addfont(str(font_path))
 
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