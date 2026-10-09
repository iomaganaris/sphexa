#!/usr/bin/env python3
"""
Plot the distribution of particles to ranks from SPH-EXA coords_rank*.txt files.

Produces:
  - A static PNG 3D scatter (matplotlib) with domain box edges drawn to scale
  - An interactive HTML 3D scatter (plotly) with pan/zoom/hover

Usage:
    python plot_rank_distribution.py [DIR] [--subsample N] [--png OUT] [--html OUT]

    DIR        Directory containing coords_rank*.txt (default: noah_disk_PM4_BT0_RLH5)
    --subsample  Number of particles to plot per rank (0 = all, default: 50000)
    --png       Output PNG filename (default: rank_distribution.png)
    --html      Output HTML filename (default: rank_distribution.html)
"""

import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import plotly.graph_objects as go


# Print-friendly colors for up to 16 ranks (Paul Tol's palettes plus a few darker tones): no pale tints, every
# color >= 2.8:1 against white paper. The order keeps consecutive ranks (often spatial neighbours) apart:
# OKLab distance >= 12.7 (x100) for ranks r, r+1 and >= 7.7 for r, r+2, also under simulated protan-, deutan-
# and tritanopia.
PRINT_PALETTE = [
    "#EE3377", "#6699CC", "#B8860B", "#332288", "#009988", "#882255", "#999933", "#117733",
    "#AA4499", "#44AA99", "#EE7733", "#0077BB", "#555555", "#CC6677", "#1F4E79", "#CC3311",
]


def rank_color(rank, num_ranks):
    """Color of a rank: PRINT_PALETTE for up to 16 ranks, otherwise a generated hue.

    Adjacent ranks are placed at opposite sides of the HSV color wheel by
    stepping the hue with the golden ratio (~0.618). This guarantees that
    rank N and rank N+1 always differ by ~137 degrees of hue, making them
    trivially distinguishable even in a dense scatter.

    For > 256 ranks, saturation and value are cycled in a second pass so
    that ranks 256 apart (which would otherwise repeat a hue) still differ.
    """
    if num_ranks <= len(PRINT_PALETTE):
        return PRINT_PALETTE[rank]

    import colorsys

    golden_conj = 0.61803398875  # 1/phi, gives ~137.5 deg steps

    if num_ranks <= 256:
        hue = (rank * golden_conj) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    else:
        hue = ((rank % 256) * golden_conj) % 1.0
        sv_idx = rank // 256
        sats = [0.85, 0.55, 0.95, 0.70, 0.40, 0.65, 0.30, 0.90]
        vals = [0.95, 0.70, 0.50, 0.85, 0.60, 0.40, 0.75, 0.30]
        s = sats[sv_idx % len(sats)]
        v = vals[(sv_idx // len(sats)) % len(vals)]
        r, g, b = colorsys.hsv_to_rgb(hue, s, v)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def load_coords(data_dir):
    """Read all coords_rank*.txt files. Returns (positions, ranks, domain_box)."""
    pattern = os.path.join(data_dir, "coords_rank*.txt")
    files = sorted(glob.glob(pattern), key=lambda f: int(
        os.path.basename(f).replace("coords_rank", "").replace(".txt", "")))

    if not files:
        sys.exit(f"No coords_rank*.txt files found in {data_dir}")

    all_pos = []
    all_ranks = []

    for f in files:
        rank = int(os.path.basename(f).replace("coords_rank", "").replace(".txt", ""))
        data = np.loadtxt(f)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        all_pos.append(data)
        all_ranks.append(np.full(len(data), rank, dtype=np.int32))
        print(f"  rank {rank:2d}: {len(data):,} particles")

    positions = np.vstack(all_pos)
    ranks = np.concatenate(all_ranks)

    # Domain extents from the full particle set
    box = np.array([
        positions[:, 0].min(), positions[:, 0].max(),
        positions[:, 1].min(), positions[:, 1].max(),
        positions[:, 2].min(), positions[:, 2].max(),
    ])

    print(f"\n  Total particles: {len(positions):,}")
    print(f"  Domain: x[{box[0]:.4f}, {box[1]:.4f}]  lx={box[1]-box[0]:.4f}")
    print(f"          y[{box[2]:.4f}, {box[3]:.4f}]  ly={box[3]-box[2]:.4f}")
    print(f"          z[{box[4]:.4f}, {box[5]:.4f}]  lz={box[5]-box[4]:.4f}")

    return positions, ranks, box, files


def subsample(positions, ranks, n_per_rank, seed=42):
    """Take at most n_per_rank particles from each rank (random uniform)."""
    if n_per_rank <= 0:
        return positions, ranks

    rng = np.random.default_rng(seed)
    mask = np.zeros(len(positions), dtype=bool)

    for r in np.unique(ranks):
        idx = np.where(ranks == r)[0]
        if len(idx) > n_per_rank:
            chosen = rng.choice(idx, size=n_per_rank, replace=False)
        else:
            chosen = idx
        mask[chosen] = True

    n_total = mask.sum()
    print(f"  Subsampled to {n_total:,} particles (max {n_per_rank:,} per rank)")

    return positions[mask], ranks[mask]


def visible_box_edges(elev, azim):
    """Box edges facing the viewer, for the current camera angle.

    The particle cloud fills the domain, so edges on the far side of the box are
    physically hidden behind it. For a convex box an edge is visible iff at least
    one of its two adjacent faces is front-facing, i.e. its outward normal has a
    positive component along the direction from the scene to the camera. A generic
    angle leaves 9 of the 12 edges visible.

    Returns the visible edges as ((axis, a, b), ...) where axis is the direction the
    edge runs along and a/b are the 0=min, 1=max choices on the other two axes.
    """
    e, a = np.radians(elev), np.radians(azim)
    view = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])

    def front_facing(axis, side):
        """side 0 = min face (outward normal -1), 1 = max face (normal +1)."""
        return view[axis] * (1.0 if side else -1.0) > 0

    out = []
    for axis in range(3):
        # the two axes the edge is pinned on, in ascending order
        o1, o2 = [k for k in range(3) if k != axis]
        for s1 in (0, 1):
            for s2 in (0, 1):
                if front_facing(o1, s1) or front_facing(o2, s2):
                    out.append((axis, s1, s2))
    return out


def title_text(positions, n_total, label=None):
    """Title carrying the run's label (e.g. "Hybrid-Dimensional SFC"), the total particle count, plus how many are plotted."""
    n_shown = len(positions)
    head = "Particle distribution to ranks" + (f", {label}" if label else "")
    if n_total is None or n_total == n_shown:
        return f"{head} ({n_shown:,} particles)"
    return f"{head} ({n_total:,} particles total, {n_shown:,} shown)"


def drawn_bbox(fig):
    """Figure-fraction bbox of everything drawn so far, from the rendered pixels."""
    fig.canvas.draw()
    ink = (np.asarray(fig.canvas.buffer_rgba())[..., :3] < 235).any(axis=2)
    rows, cols = np.where(ink.any(axis=1))[0], np.where(ink.any(axis=0))[0]
    h, w = ink.shape
    return Bbox.from_extents(cols[0] / w, 1 - (rows[-1] + 1) / h, (cols[-1] + 1) / w, 1 - rows[0] / h)


def plot_static(positions, ranks, box, out_png, num_ranks, n_total=None, label=None, size=0.3, alpha=0.5,
                section_y=None, section_at=None):
    """3D scatter with matplotlib, axes scaled to domain, box edges drawn. With section_y (a half width), an x-z cross
    section at y = section_at (default: the middle of the box in y; particles within section_y of it) is drawn below,
    at true scale."""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Only the front edges are drawn, so they must render over the scatter rather
    # than be depth-sorted against it as one lump.
    ax.computed_zorder = False

    lx, ly, lz = box[1] - box[0], box[3] - box[2], box[5] - box[4]

    # One scatter for all ranks, coloured per particle: matplotlib depth-sorts the points of a 3D scatter, so the
    # particles nearest the camera are drawn last whichever rank they belong to (one scatter per rank would paint
    # the later ranks over the earlier ones regardless of depth).
    colors = np.array([matplotlib.colors.to_rgba(rank_color(r, num_ranks)) for r in range(num_ranks)])[ranks]
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c=colors, s=size, alpha=alpha, rasterized=True,
               zorder=1)

    # Draw the domain box edges that face the viewer; the ones behind the cloud
    # would be hidden by the particles anyway.
    lo = np.array([box[0], box[2], box[4]])
    hi = np.array([box[1], box[3], box[5]])
    for axis, s1, s2 in visible_box_edges(ax.elev, ax.azim):
        o1, o2 = [k for k in range(3) if k != axis]
        start, end = np.empty(3), np.empty(3)
        start[o1] = end[o1] = hi[o1] if s1 else lo[o1]
        start[o2] = end[o2] = hi[o2] if s2 else lo[o2]
        start[axis], end[axis] = lo[axis], hi[axis]
        ax.plot(*zip(start, end), color="black", linewidth=0.8, alpha=0.6, zorder=10)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    # Equal aspect ratio: scale each axis so the box proportions are correct
    ax.set_box_aspect((lx, ly, lz))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    # Legend and title are placed around what the 3D axes actually draw (cube, ticks, axis labels), which is much
    # smaller than the axes area (and than ax.get_tightbbox), so it is measured on the rendered pixels; the saved
    # figure is then cropped to plot, legend and title.
    plot_box = drawn_bbox(fig)
    pt_per_fig = fig.get_size_inches() * 72  # figure fraction -> points, (x, y)

    section = None
    if section_y is not None:
        # Grow the figure downwards by the section panel (keeping the 3D axes' size in inches, so it renders as
        # before) and put the panel under the drawn plot, as wide as it, with x:z at true scale.
        w_in, h_in = fig.get_size_inches()
        panel_w = plot_box.width * w_in
        panel_h = panel_w * lz / lx
        title_gap, label_room = 34 / 72, 30 / 72  # space for the panel title above, x ticks and label below
        extra = max(0.0, title_gap + panel_h + label_room - plot_box.y0 * h_in)
        fig.set_size_inches(w_in, h_in + extra)
        ax.set_position([0, extra / (h_in + extra), 1, h_in / (h_in + extra)])
        plot_box = drawn_bbox(fig)
        pt_per_fig = fig.get_size_inches() * 72
        H = h_in + extra

        y_mid = (box[2] + box[3]) / 2 if section_at is None else section_at
        m = np.abs(positions[:, 1] - y_mid) < section_y
        order = np.argsort(-positions[m, 1])  # viewer at -y: the lowest-y particles are nearest, drawn last
        sec_pos, sec_col = positions[m][order], colors[m][order]
        section = fig.add_axes([plot_box.x0, plot_box.y0 - title_gap / H - panel_h / H, plot_box.width, panel_h / H])
        section.scatter(sec_pos[:, 0], sec_pos[:, 2], c=sec_col, s=size, alpha=alpha, rasterized=True)
        section.set_xlim(box[0], box[1])
        section.set_ylim(box[4], box[5])
        section.set_aspect("equal", adjustable="box")
        section.set_xlabel("X")
        section.set_ylabel("Z")
        section.set_title(f"Cross section at y = {y_mid:.1f} (±{section_y:g}, {m.sum():,} particles)",
                          fontsize=14)

    gap = 12 / pt_per_fig[0]

    # legend: one column left of the plot, spaced to be as tall as it, with opaque markers (the scatter's are
    # tiny and translucent)
    shown = [r for r in range(num_ranks) if (ranks == r).any()]
    handles = [plt.Line2D([], [], linestyle="", marker="o", markersize=8, color=rank_color(r, num_ranks))
               for r in shown]
    fontsize = 12
    height = plot_box.height * pt_per_fig[1]
    spacing = max(0.3, (height - len(shown) * fontsize) / (max(len(shown) - 1, 1) * fontsize))
    legend = fig.legend(handles, [f"rank {r}" for r in shown], loc="center right",
               bbox_to_anchor=(plot_box.x0 - gap, (plot_box.y0 + plot_box.y1) / 2), ncol=1, frameon=False,
               fontsize=fontsize, handletextpad=0.3, labelspacing=spacing, borderpad=0, borderaxespad=0)

    title = fig.text((plot_box.x0 + plot_box.x1) / 2, plot_box.y1 + 6 / pt_per_fig[1],
                     title_text(positions, n_total, label), ha="center", va="bottom", fontsize=17)

    renderer = fig.canvas.get_renderer()
    # a title wider than the plot is kept over plot and legend: shifted left where it would stick out on the right,
    # centred on both if it is wider than them, so it does not widen the cropped figure
    to_fig = fig.transFigure.inverted()
    tb = title.get_window_extent(renderer).transformed(to_fig)
    lb = legend.get_window_extent(renderer).transformed(to_fig)
    left, right = min(lb.x0, plot_box.x0), plot_box.x1
    if tb.width >= right - left:
        title.set_x((left + right) / 2)
    elif tb.x1 > right:
        title.set_x(right - tb.width / 2)
    # the crop below measures the title where it is now, at the saved resolution (text is wider at the screen's)
    fig.set_dpi(200)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_inches = fig.dpi_scale_trans.inverted()
    content = Bbox.union([plot_box.transformed(fig.transFigure).transformed(to_inches),
                          legend.get_window_extent(renderer).transformed(to_inches),
                          title.get_window_extent(renderer).transformed(to_inches)]
                         + ([section.get_tightbbox(renderer).transformed(to_inches)] if section else []))
    fig.savefig(out_png, dpi=200, bbox_inches=content.padded(0.1))
    plt.close(fig)
    print(f"  Saved static plot: {out_png}")


def plot_interactive(positions, ranks, box, out_html, num_ranks, n_total=None, label=None, size=1.2, opacity=0.5):
    """Interactive full-screen 3D scatter with plotly, axes scaled to domain."""
    lx, ly, lz = box[1] - box[0], box[3] - box[2], box[5] - box[4]

    traces = []
    for r in range(num_ranks):
        mask = ranks == r
        if not mask.any():
            continue
        color = rank_color(r, num_ranks)
        traces.append(go.Scatter3d(
            x=positions[mask, 0], y=positions[mask, 1], z=positions[mask, 2],
            mode="markers",
            marker=dict(size=size, color=color, opacity=opacity),
            name=f"rank {r}",
        ))

    # Domain box edges as a single trace
    xmin, xmax, ymin, ymax, zmin, zmax = box
    c = [
        [xmin, ymin, zmin], [xmax, ymin, zmin], [xmax, ymax, zmin],
        [xmin, ymax, zmin], [xmin, ymin, zmin], [xmin, ymin, zmax],
        [xmax, ymin, zmax], [xmax, ymax, zmax], [xmin, ymax, zmax],
        [xmin, ymin, zmax], [xmin, ymin, zmin],  # close bottom-left vertical
        [xmin, ymax, zmin], [xmin, ymax, zmax],
        [xmax, ymin, zmin], [xmax, ymin, zmax],
        [xmax, ymax, zmin], [xmax, ymax, zmax],
    ]
    edges = np.array(c)
    traces.append(go.Scatter3d(
        x=edges[:, 0], y=edges[:, 1], z=edges[:, 2],
        mode="lines",
        line=dict(color="black", width=2),
        name="domain box",
    ))

    fig = go.Figure(data=traces)

    fig.update_layout(
        title=dict(text=title_text(positions, n_total, label), font=dict(size=22), x=0.5, xanchor="center"),
        scene=dict(
            domain=dict(x=[0.1, 1]),  # leave the left column to the legend
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="manual",
            aspectratio=dict(x=lx / max(lx, ly, lz),
                             y=ly / max(lx, ly, lz),
                             z=lz / max(lx, ly, lz)),
            xaxis=dict(range=[box[0], box[1]]),
            yaxis=dict(range=[box[2], box[3]]),
            zaxis=dict(range=[box[4], box[5]]),
        ),
        legend=dict(font=dict(size=13), x=0, xanchor="left", y=0.5, yanchor="middle", itemsizing="constant"),
        paper_bgcolor="white",
        margin=dict(l=0, r=0, t=40, b=0),
    )

    # Generate the HTML with a custom wrapper that fills the viewport.
    # Plotly's responsive config + a 100vw/100vh div makes the plot scale
    # to the full browser window on any screen size.
    inner_html = fig.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        div_id="plot-div",
        config={"responsive": True},
    )

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        html, body {{
            margin: 0;
            padding: 0;
            width: 100%;
            height: 100%;
            overflow: hidden;
        }}
        #plot-div {{
            width: 100vw;
            height: 100vh;
        }}
    </style>
</head>
<body>
{inner_html}
</body>
</html>"""

    with open(out_html, "w") as f:
        f.write(full_html)

    print(f"  Saved interactive plot: {out_html}")


def main():
    parser = argparse.ArgumentParser(description="Plot particle-to-rank distribution from coords_rank*.txt")
    parser.add_argument("dir", nargs="?", default="noah_disk_PM4_BT0_RLH5",
                        help="Directory containing coords_rank*.txt (default: noah_disk_PM4_BT0_RLH5)")
    parser.add_argument("--subsample", type=int, default=50000,
                        help="Max particles per rank to plot (0 = all, default: 50000)")
    parser.add_argument("--png", default="rank_distribution.png",
                        help="Output PNG filename (default: rank_distribution.png)")
    parser.add_argument("--html", default="rank_distribution.html",
                        help="Output HTML filename (default: rank_distribution.html)")
    args = parser.parse_args()

    repo = "/capstor/scratch/cscs/ioannmag/CORNERSTONE/sphexa"
    data_dir = args.dir if os.path.isabs(args.dir) else os.path.join(repo, args.dir)

    print(f"Loading coords from {data_dir} ...")
    positions, ranks, box, files = load_coords(data_dir)
    num_ranks = len(files)
    n_total = len(positions)

    print(f"\nSubsampling (max {args.subsample:,} per rank) ...")
    sub_pos, sub_ranks = subsample(positions, ranks, args.subsample)

    # Free full data before plotting
    del positions, ranks

    print(f"\nGenerating static PNG ...")
    plot_static(sub_pos, sub_ranks, box, args.png, num_ranks, n_total)

    print(f"\nGenerating interactive HTML ...")
    plot_interactive(sub_pos, sub_ranks, box, args.html, num_ranks, n_total)

    print("\nDone.")


if __name__ == "__main__":
    main()
