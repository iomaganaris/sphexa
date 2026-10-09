#!/usr/bin/env python3
"""Share of the step time spent per phase group, from the per-rank timings in profile.h5.

Usage: plot_phase_groups.py [-o OUT_STEM] [--title T] VARIANT@RANKS=RUN_DIR[:NOPROF_DIR] [...]

  * per run: per-rank sums over the logged iterations, skipping the first FIRST_STEP (as compare_runs_log.py),
    then averaged over ranks; percentages are group time / total step time
  * the rank-0 split is printed as well, as a cross-check against the log-based results files
  * writes OUT_STEM.txt (table) and OUT_STEM.png (100% stacked bars grouped by rank count)
  * --runtime-heights writes OUT_STEM_runtime.png instead: bar height = runtime, the max over ranks of the step
    timers summed over the counted steps (the ranks synchronise every step, so this equals rank 0's runtime and
    the log's summed 'Total time for iteration'), split by the same group shares; nonmixd-BT0 is drawn in its own
    graph next to BT3 and mixd-BT0 so that every bar keeps its true height on a linear axis; speedups
    (reference runtime / runtime) are shown as a bracket for mixd-BT0 vs BT3 and as a row under each group for
    BT3 and mixd-BT0 vs nonmixd-BT0
  * --variants nonmixd-BT0,mixd-BT0 plots a subset (labels drop -BT0 when no BT3 run is shown); with two
    variants both share one graph with a speedup bracket per group; a bar of the first variant more than
    CLIP_RATIO x taller than every bar of the others (as in plot_bt_scaling.py) is cut once inside its largest
    segment: the cut ('//') takes out the time written next to it, everything else stays on the one linear scale
    (the y-axis holds below the cuts, the runtime above the bar is the true one); --split-axis instead splits the
    y-axis: the lower part shows 0..SPLIT_HEADROOM x the shorter bars, the upper part the rest on a coarser scale
  * --log (with --runtime-heights) writes OUT_STEM_runtime_log.png instead: log y-axis from the power of 10 below
    the shortest bar; the bar top is the runtime, the drawn bar is split by the group shares in proportion to its
    drawn (log) length, so inner segment edges are not readable on the axis
  * --sync-group moves Timestep and duTimestep into a group of their own, "Waiting for the slowest rank", and
    writes OUT_STEM_sync*: the observables' MPI_Reduce (between computeForces and integrate) lands in rank 0's
    duTimestep, the other ranks wait in Timestep's global reduction; both also hold a little real work (the
    time-step minima), so the group is an upper bound of the wait
  * --absolute writes OUT_STEM_absolute.png instead: one panel per phase group, in each a pair of bars per rank
    count (one bar per variant) with the group's time in seconds (summed over the counted steps, mean over
    ranks) on a log y-axis shared by all panels, so every bar is read directly off the axis; the factor above a
    pair is first variant / second variant (> 1: the second is faster)
  * --narrow (with --runtime-heights or --absolute) writes OUT_STEM_runtime[_log]_narrow.png instead: a half-page-wide figure
    with larger fonts and the header wrapped over several lines
  * with :NOPROF_DIR the runtime is taken from NOPROF_DIR/run.log instead (rank-0 'Total time for iteration'
    summed over the same counted steps, from a run without --profile); the shares still come from RUN_DIR
"""

import argparse
import math
import re
import textwrap

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compare_runs_log import FIRST_STEP, parse_run
from plot_bt_comparison import GRID, HALO_COLORS, HALO_EDGE, TEXT, TEXT2, VARIANTS
from plot_bt_scaling import CLIP_RATIO, label

GROUPS = {
    "Domain decomposition and communication": [
        "domain::sync", "computeGroups", "updateSmoothingLengthIterative", "FindNeighbors",
        "mpi::synchronizeHalos", "Upsweep", "computeAndExchangeStarPosition", "exchangeAndAccreteOnStar",
        "Timestep",
    ],
    "Gravity": ["Gravity"],
    "Particle interaction": None,  # every other phase
}
# navy / light orange / green: print grey 59 / 178 / 116, so the groups stay apart in black and white
GROUP_COLORS = ["#1c3f74", "#f4a259", "#2e9a68"]
GROUP_COLOR = dict(zip(GROUPS, GROUP_COLORS))  # fixed per group, also when an empty group is dropped
SYNC_GROUP = "Waiting for the slowest rank"  # --sync-group: drawn on top of the bar
SYNC_PHASES = ["Timestep", "duTimestep"]
SYNC_COLOR = "#e3e1dc"  # print grey 225, hatched (SYNC_HATCH) to stay apart from Gravity in black and white
SYNC_HATCH, SYNC_HATCH_COLOR = "///", "#a9a7a1"
INK = {"#1c3f74": "white", "#2e9a68": "white"}  # text colour inside a segment (default TEXT)
LABEL_INK = {"#f4a259": "#b8641e", SYNC_COLOR: TEXT2}  # share labels above a bar: darker shade of a light segment
SPLIT_HEADROOM = 1.4  # lower part of a split y-axis: up to this x the tallest of the shorter bars
SPLIT_HEIGHTS = [1, 1.6]  # upper, lower part of a split y-axis
CUT_KEEP = 0.35  # a cut bar keeps this x the tallest uncut bar (or uncut part) of its largest segment
# layout of the graph, changed by --narrow: font = scale of the label fonts, step / split_step = scale of the vertical
# gaps between the labels stacked on a bar (on a single / split y-axis), spacing / bar = bar pitch / width within a group, gap = extra space between
# groups, rank_y / vs_y = axes-fraction y of the 'ranks' row and the 'vs 3D SFC' row under the bars, tick / rank =
# font size of the variant / rank labels, rank_fmt = rank label, ylabel_x = axes-fraction x of a split axis' y label,
# even_gaps = on a split axis, space the labels in the shorter upper part as far apart (in inches) as in the lower,
# short_ylabel = keep only the first line of the y label (main() moves the rest into the note above the graph),
# split_heights = upper, lower part of a split y-axis, variant_title = name the variants above the graph,
# pct = font size of the share labels, pct_va = their vertical alignment inside a segment, rank_pts = if set, put the
# 'ranks' row this many points below the x axis instead of at rank_y (keeps it under the tick labels at any height; moved
# down by the extra lines of wrapped tick labels), tick_wrap = width in characters the variant tick labels wrap at, cut_at = height of the // cut inside a cut bar's
# segment (fraction of the segment; its share label sits halfway below it, the removed time halfway above)
STYLE = dict(font=1.0, step=1.0, split_step=1.8, spacing=0.3, bar=0.26, gap=0.35, rank_y=-0.17, vs_y=-0.25, tick=8.5, rank=10,
             rank_fmt="{} ranks (GPUs)", ylabel_x=-0.06, even_gaps=True, short_ylabel=False,
             split_heights=SPLIT_HEIGHTS, variant_title=True, pct=8, pct_va="center",
             rank_pts=None, tick_wrap=17, cut_at=0.5)
NARROW = dict(font=1.3, step=2.2, split_step=2.2, spacing=0.46, bar=0.32, gap=0.07, rank_y=-0.13, vs_y=-0.29, tick=9, rank=11.5,
              rank_fmt="{} ranks", ylabel_x=-0.13, even_gaps=True, short_ylabel=True, split_heights=[1, 1],
              variant_title=False, pct=9.5, pct_va="center_baseline",
              rank_pts=21, tick_wrap=11, cut_at=0.4)


def group_of(phase):
    for g, members in GROUPS.items():
        if members is not None and phase in members:
            return g
    return next(g for g, m in GROUPS.items() if m is None)


def load(run_dir, noprof_dir=None):
    """Seconds per group summed over the counted steps: averaged over ranks ("mean"), rank 0 ("rank0") and the
    max over ranks ("max", per group, so these do not add up to the runtime); plus the runtime and step count."""
    with h5py.File(f"{run_dir}/profile.h5", "r") as f:
        g = f["Step#0"]
        a = g.attrs
        nr, ni, nd = (int(a[k][0]) for k in ("numRanks", "numIterations", "numDataPerIteration"))
        names = [a[f"stepName{i}"] for i in range(nd)]
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in names]
        t = g["timings"][:].reshape(nr, ni, nd)[:, FIRST_STEP:, :]  # [rank][iteration][phase]
    per_rank = t.sum(1)  # [rank][phase]
    out = {}
    per_group = {gname: per_rank[:, [i for i, n in enumerate(names) if group_of(n) == gname]].sum(1)
                 for gname in GROUPS}  # [rank] per group
    for which, reduce in (("mean", np.mean), ("rank0", lambda v: v[0]), ("max", np.max)):
        out[which] = {gname: float(reduce(v)) for gname, v in per_group.items()}
    out["steps"] = t.shape[1]
    # max over ranks; the ranks synchronise every step, so this equals rank 0 and the log's summed
    # 'Total time for iteration'
    out["runtime"] = per_rank.sum(1).max()
    out["runtime_rank0"] = per_rank[0].sum()
    if noprof_dir:
        its = parse_run(f"{noprof_dir}/run.log", 0)["its"][FIRST_STEP:]
        assert len(its) == out["steps"], f"{noprof_dir}: {len(its)} counted steps, profile has {out['steps']}"
        out["runtime"] = out["runtime_rank0"] = sum(i[1] for i in its)  # the log only has rank 0
    return out


def draw(axs, data, ranks, variants, pctg, absolute, steps_txt, split=None, log=None, cuts=None):
    """Stacked bars grouped by rank count; height 100% or the rank-0 runtime, split by the group shares.

    axs = [ax], or with split = [lower ax (0..split), upper ax (split..top)]: bars are drawn on both, every
    label on the part that holds its position, spaced in that part's own units.
    log = axis floor [s]: heights are drawn as log10(runtime / log) on a linear axis whose ticks are labelled in
    seconds, so the shares split the drawn (log) length of each bar.
    cuts = {(rank, variant): (group, seconds)}: that bar is drawn shorter by `seconds`, taken out of `group`."""
    cuts = cuts or {}
    name = lambda v: label(v, VARIANTS)
    n = len(variants)
    spacing = STYLE["spacing"]
    offsets = [(k - (n - 1) / 2) * spacing for k in range(n)]
    pitch = STYLE["gap"] + n * spacing
    drawn = lambda rt: math.log10(rt / log) if log else rt
    top = max(drawn(data[(r, v)]["runtime"]) - cuts.get((r, v), (None, 0))[1]
              for r in ranks for v in variants if (r, v) in data) if absolute else 100.0
    upper_span = 1.12 * top - split if split else None
    at = lambda y: axs[1] if split and y >= split else axs[0]  # part holding height y
    step = STYLE["split_step"] if split else STYLE["step"]
    upper_unit = upper_span * STYLE["split_heights"][1] / STYLE["split_heights"][0] if split and STYLE["even_gaps"] else upper_span
    unit = lambda y: (upper_unit if y >= split else split) if split else top  # its span, for label spacing

    def line(x0, y0, x1, y1):  # horizontal or vertical; a vertical one crossing the split is drawn on both parts
        pieces = [(y0, y1)]
        if split and x0 == x1 and min(y0, y1) < split < max(y0, y1):
            pieces = [(min(y0, y1), split), (split, max(y0, y1))]
        for a, b in pieces:
            at((a + b) / 2).plot([x0, x1], [a, b], color=TEXT2, lw=1, zorder=2)

    xs, xl = [], []
    wrap = lambda v: textwrap.wrap(name(v), STYLE["tick_wrap"], break_long_words=False)  # never splits a word
    tick_lines = max(len(wrap(v)) for v in variants) if n > 1 else 1
    stack_top = {}  # (rank, variant) -> (x, y above the bar's labels)
    ymax_needed = top
    for gi, r in enumerate(ranks):
        for off, v in zip(offsets, variants):
            if (r, v) not in data:
                continue
            x = gi * pitch + off
            xs.append(x); xl.append("\n".join(wrap(v)) if n > 1 else f"{r} ranks\n(GPUs)")
            p = pctg(data[(r, v)]["mean"])
            rt = data[(r, v)]["runtime"]
            full = drawn(rt) if absolute else 100.0
            cut_group, removed = cuts.get((r, v), (None, 0))
            height = full - removed
            bottom = 0.0
            above = []  # segments too thin for a label inside: labelled on top of the bar in their own colour
            for g, share in p.items():
                col = GROUP_COLOR[g]
                val = share * full / 100 - (removed if g == cut_group else 0)
                for k, ax in enumerate(axs):
                    hatch = dict(hatch=SYNC_HATCH, hatchcolor=SYNC_HATCH_COLOR) if g == SYNC_GROUP else {}
                    ax.bar(x, val, bottom=bottom, width=STYLE["bar"], color=col, edgecolor="white", linewidth=1.5,
                           zorder=2, **hatch, label=g if (k, gi, v) == (0, 0, variants[0]) else None)
                # label where the segment shows the most of its height: (fraction of that part's span, mid y)
                parts = [(val / top, bottom + val / 2)]
                if split:
                    lo_a, lo_b = min(bottom, split), min(bottom + val, split)
                    up_a, up_b = max(bottom, split), max(bottom + val, split)
                    parts = [((lo_b - lo_a) / split, (lo_a + lo_b) / 2), ((up_b - up_a) / upper_span, (up_a + up_b) / 2)]
                frac, ymid = max(parts)
                if g == cut_group:  # share below the cut, the removed time above it
                    c = STYLE["cut_at"]
                    ymid = bottom + c * val / 2
                    yb, w = bottom + c * val, STYLE["bar"]
                    for dy in (-0.008 * top, 0.008 * top):
                        axs[0].plot([x - w / 2, x + w / 2], [yb + dy - 0.006 * top, yb + dy + 0.006 * top],
                                    color=TEXT2, lw=2.5, zorder=3, solid_capstyle="butt")
                    axs[0].text(x, bottom + (1 + c) * val / 2, f"−{removed:,.0f} s", rotation=90, ha="center",
                                va="center", fontsize=0.9 * STYLE["pct"], color=TEXT2, zorder=3)
                if frac >= 0.04:
                    at(ymid).text(x, ymid, f"{share:.0f}%", ha="center", va=STYLE["pct_va"], fontsize=STYLE["pct"],
                                  color=INK.get(col, TEXT),
                                  zorder=3)
                else:
                    above.append((share, col))
                bottom += val
            y = height + 0.012 * step * unit(height)
            for share, col in above:  # stack order: lowest segment's label lowest
                at(y).text(x, y, f"{share:.0f}%", ha="center", va="bottom", fontsize=STYLE["pct"], color=LABEL_INK.get(col, col),
                           fontweight="bold")
                y += 0.045 * step * unit(y)
            if absolute:
                at(y).text(x, y, f"{rt:,.0f} s", ha="center", va="bottom", fontsize=8 * STYLE["font"], color=TEXT)
                y += 0.045 * step * unit(y)
            stack_top[(r, v)] = (x, y)
            ymax_needed = max(ymax_needed, y)
        if n > 1:  # a single-variant graph labels its bars by rank count directly (the title names the variant)
            xr = gi * pitch + (offsets[0] + offsets[-1]) / 2
            if STYLE["rank_pts"]:
                axs[0].annotate(STYLE["rank_fmt"].format(r), (xr, 0), xycoords=axs[0].get_xaxis_transform(),
                                xytext=(0, -STYLE["rank_pts"] - (tick_lines - 1) * 1.2 * STYLE["tick"]), textcoords="offset points", ha="center", va="top",
                                fontsize=STYLE["rank"], color=TEXT)
            else:
                axs[0].text(xr, STYLE["rank_y"], STYLE["rank_fmt"].format(r), transform=axs[0].get_xaxis_transform(),
                            ha="center", va="top", fontsize=STYLE["rank"], color=TEXT)
        if absolute and n == 2 and all((r, v) in data for v in variants):  # speedup of the 2nd vs the 1st
            ref, case = variants
            (xa, ya), (xb, yb) = stack_top[(r, ref)], stack_top[(r, case)]
            y = max(ya, yb) + 0.01 * step * unit(max(ya, yb))
            h = 0.02 * step * unit(y)
            line(xa, ya - 0.005 * unit(ya), xa, y); line(xa, y, xb, y); line(xb, y, xb, yb - 0.005 * unit(yb))
            speedup = data[(r, ref)]["runtime"] / data[(r, case)]["runtime"]
            at(y).text((xa + xb) / 2, y + 0.008 * step * unit(y), f"{speedup:.2f}×", ha="center", va="bottom", fontsize=8 * STYLE["font"],
                       color=TEXT)
            ymax_needed = max(ymax_needed, y + 0.05 * step * unit(y))
        if absolute and "nonmixd-BT0" not in variants and (r, "nonmixd-BT0") in data:
            ref = data[(r, "nonmixd-BT0")]["runtime"]
            for v in variants:
                if (r, v) in data:
                    axs[0].text(stack_top[(r, v)][0], STYLE["vs_y"], f"{ref / data[(r, v)]['runtime']:.2f}×",
                                transform=axs[0].get_xaxis_transform(), ha="center", va="top", fontsize=8.5 * STYLE["font"],
                                color=TEXT)
            if gi == 0:
                axs[0].text(-0.01, STYLE["vs_y"], f"vs {name('nonmixd-BT0')}:", transform=axs[0].transAxes, ha="right",
                            va="top", fontsize=8.5 * STYLE["font"], color=TEXT2)
    for ax in axs:
        ax.set_xticks(xs)
        ax.yaxis.grid(True, color=GRID, zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(TEXT2)
        ax.tick_params(colors=TEXT2, labelsize=10 * STYLE["font"])
    axs[0].set_xticklabels(xl, fontsize=STYLE["tick"] if n > 1 else 9.5 * STYLE["font"], color=TEXT2 if n > 1 else TEXT)
    if log:
        ylabel = (f"Runtime over {steps_txt} [s], max over ranks, log scale\nsegments: share of step time (mean over ranks),"
                  "\nas a share of the drawn bar (only bar tops read on the axis)")
        if STYLE["variant_title"]:
            axs[-1].set_title(" and ".join(map(name, variants)), fontsize=11 * STYLE["font"], color=TEXT)
        ticks = [m * 10 ** e for e in range(round(math.log10(log)), 7) for m in (1, 2, 5)]
        ticks = [t for t in ticks if drawn(t) <= 1.3 * top]
        for ax in axs:
            ax.set_yticks([drawn(t) for t in ticks])
            ax.set_yticklabels([f"{t:,.0f}" for t in ticks])
    elif absolute:
        ylabel = f"Runtime over {steps_txt} [s], max over ranks\nsegments: share of step time (mean over ranks)"
        if STYLE["variant_title"]:
            axs[-1].set_title(" and ".join(map(name, variants)), fontsize=11 * STYLE["font"], color=TEXT)
    else:
        ylabel = "Share of step time [%] (mean over ranks)"
    if STYLE["short_ylabel"]:
        ylabel = ylabel.split("\n")[0]
    if split:
        axs[0].set_ylim(0, split)
        axs[1].set_ylim(split, max(top * 1.08, ymax_needed + 0.01 * upper_span))
        axs[1].spines["bottom"].set_visible(False)
        axs[1].tick_params(axis="x", bottom=False, labelbottom=False)
        for ax, y in ((axs[0], 1), (axs[1], 0)):  # break marks on the y-axis where the scale changes
            ax.plot([-0.012, 0.012], [y - 0.012, y + 0.012], transform=ax.transAxes, color=TEXT2, lw=1,
                    clip_on=False)
        axs[1].set_ylabel(ylabel + ("" if STYLE["short_ylabel"] else "\n(upper part on a coarser scale)"), color=TEXT2,
                          fontsize=10 * min(STYLE["font"], 1.1))
        axs[1].yaxis.set_label_coords(STYLE["ylabel_x"], (1 - STYLE["split_heights"][1] / STYLE["split_heights"][0]) / 2)  # centred on both parts
    elif absolute:
        axs[0].set_ylim(0, max(top * 1.08, ymax_needed + 0.01 * top))
        axs[0].set_ylabel(ylabel, color=TEXT2, fontsize=10 * min(STYLE["font"], 1.1))
    else:
        axs[0].set_ylim(0, max(100, ymax_needed + 1))
        axs[0].set_ylabel(ylabel, color=TEXT2, fontsize=10 * min(STYLE["font"], 1.1))


def stack_header(fig, title, handles, labels, ncol, note, narrow, h_pad=None, w_pad=None):
    """Title, legend and note stacked top-down, each placed under the previous one as drawn, then the graphs fitted
    right below them."""
    r = fig.canvas.get_renderer()
    below = lambda a, gap=0.06: (a.get_window_extent(r).y0 - gap * fig.dpi) / fig.bbox.height
    w = fig.get_figwidth()  # wrap widths in characters, scaled to the figure width
    sup = fig.suptitle(textwrap.fill(title, 48 if narrow else int(8.5 * w)), y=1 - 0.08 / fig.get_figheight(), va="top",
                       fontsize=13 if narrow else 12, color=TEXT)
    fig.canvas.draw()
    leg = fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, below(sup)), ncol=ncol, frameon=False,
                     fontsize=11 if narrow else 9.5, labelcolor=TEXT, borderaxespad=0)
    fig.canvas.draw()
    items = [sup, leg]
    if note:
        items.append(fig.text(0.5, below(leg), textwrap.fill(note, 64 if narrow else int(13 * w)), ha="center", va="top",
                              fontsize=9.5 if narrow else 8.5, color=TEXT2))
        fig.canvas.draw()
    for a in items:
        a.set_in_layout(False)  # already above the rect top; tight_layout would reserve their height again
    fig.tight_layout(rect=(0, 0.01, 1, below(items[-1], 0.04)), pad=0.4, h_pad=h_pad, w_pad=w_pad)


def draw_absolute(data, ranks, groups, steps_txt, title, narrow):
    """One panel per phase group; per rank count a bar per variant, log y-axis in seconds shared by all panels."""
    name = lambda v: label(v, VARIANTS)
    values = [d["mean"][g] for d in data.values() for g in groups if d["mean"][g] > 0]
    floor = 10 ** math.floor(math.log10(min(values)))
    top = max(values)
    ncols = min(2, len(groups)) if narrow else len(groups)
    nrows = math.ceil(len(groups) / ncols)
    panel_w = 0.6 + 0.7 * len(ranks)
    figsize = (6, 1.9 + 3.05 * nrows) if narrow else (0.8 + panel_w * len(groups), 6.6)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=True, squeeze=False)
    fs = STYLE["font"]
    width = 0.38
    for ax in axes.flat[len(groups):]:
        ax.set_visible(False)
    for k, (ax, g) in enumerate(zip(axes.flat, groups)):
        for j, v in enumerate(VARIANTS):
            pts = [(i + (j - (len(VARIANTS) - 1) / 2) * width, data[(r, v)]["mean"][g])
                   for i, r in enumerate(ranks) if (r, v) in data]
            ax.bar([x for x, _ in pts], [y for _, y in pts], width=width, color=HALO_COLORS[v],
                   edgecolor=HALO_EDGE.get(v, "white"),
                   linewidth=1.2, zorder=2, label=name(v))
            for x, val in pts:
                if val > 0:
                    ax.annotate(f"{val:,.0f}" if val >= 10 else f"{val:.1f}", (x, val), xytext=(0, 2),
                                textcoords="offset points", rotation=90, ha="center", va="bottom",
                                fontsize=8 * fs, color=TEXT)
        if len(VARIANTS) == 2:
            a, b = VARIANTS
            for i, r in enumerate(ranks):
                if (r, a) in data and (r, b) in data:
                    va, vb = data[(r, a)]["mean"][g], data[(r, b)]["mean"][g]
                    if va > 0 and vb > 0:  # above the taller bar's rotated value label
                        digits = len(f"{max(va, vb):,.0f}" if max(va, vb) >= 10 else f"{max(va, vb):.1f}")
                        ax.annotate(f"{va / vb:.1f}×", (i, max(va, vb)), xytext=(0, 5 + 0.6 * 8 * fs * digits),
                                    textcoords="offset points", ha="center", va="bottom",
                                    fontsize=9 if narrow else 8.5, color=TEXT, fontweight="bold")
        ax.set_yscale("log")
        ax.set_title(textwrap.fill(g, 20 if narrow else 24), fontsize=12 if narrow else 11, color=TEXT)
        ax.set_xticks(range(len(ranks)))
        ax.set_xticklabels([str(r) for r in ranks])
        ax.set_xlim(-0.6, len(ranks) - 0.4)
        if k >= len(groups) - ncols:  # bottom panel of its column
            ax.set_xlabel("Ranks (GPUs)", color=TEXT2, fontsize=10 * min(fs, 1.1))
        ax.yaxis.grid(True, color=GRID, zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(TEXT2)
        ax.tick_params(colors=TEXT2, labelsize=10 * fs)
        ax.tick_params(axis="y", labelleft=True)
    # headroom above the tallest bar for its rotated value and the factor: 45% of the data span in log space
    axes[0][0].set_ylim(floor, 10 ** (math.log10(top) + 0.45 * (math.log10(top) - math.log10(floor))))
    axes[0][0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda t, _: f"{t:,.0f}" if t >= 1 else f"{t:g}"))
    for row in axes:
        row[0].set_ylabel(f"Time over {steps_txt} [s]" + ("" if narrow else ", log scale"), color=TEXT2,
                          fontsize=10 * min(fs, 1.1))
    ref, case = (name(v) for v in VARIANTS[:2])
    note = (f"Bars: time per phase group, mean over ranks, log y-axis shared by all graphs. Above each pair: "
            f"{ref} time / {case} time (> 1: {case} faster)" + ("; 1 GPU per rank" if narrow else ""))
    handles, labels = axes[0][0].get_legend_handles_labels()
    stack_header(fig, f"{title} ({steps_txt})", handles, labels, len(VARIANTS), note, narrow, h_pad=1.0,
                 w_pad=0.8)
    return fig


def main():
    global VARIANTS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="VARIANT@RANKS=RUN_DIR[:NOPROF_DIR]")
    ap.add_argument("-o", "--output", default="phase_groups", help="output stem (default: %(default)s)")
    ap.add_argument("--title", default="Protoplanetary Disk experiment: time per phase group")
    ap.add_argument("--runtime-heights", action="store_true",
                    help="bar height = runtime instead of 100%% (writes OUT_STEM_runtime.png)")
    ap.add_argument("--variants", default=",".join(VARIANTS),
                    help="comma-separated subset of %(default)s to plot (default: all)")
    ap.add_argument("--log", action="store_true",
                    help="with --runtime-heights: log y-axis, one graph (writes OUT_STEM_runtime_log.png)")
    ap.add_argument("--absolute", action="store_true",
                    help="grouped bars of the time per phase group on a log axis, one panel per phase group "
                         "(writes OUT_STEM_absolute.png)")
    ap.add_argument("--split-axis", action="store_true",
                    help="split the y-axis (coarser upper part) instead of cutting the tall bars")
    ap.add_argument("--sync-group", action="store_true",
                    help=f"move {', '.join(SYNC_PHASES)} into a group '{SYNC_GROUP}' (writes OUT_STEM_sync*)")
    ap.add_argument("--narrow", action="store_true",
                    help="with --runtime-heights: half-page-wide figure with larger fonts (writes OUT_STEM_runtime*_narrow.png)")
    args = ap.parse_args()
    assert not args.log or args.runtime_heights, "--log needs --runtime-heights"
    assert not args.narrow or args.runtime_heights or args.absolute, "--narrow needs --runtime-heights or --absolute"
    assert not (args.absolute and (args.runtime_heights or args.log)), "--absolute replaces --runtime-heights / --log"
    if args.narrow:
        STYLE.update(NARROW)
    if args.sync_group:
        for g, members in GROUPS.items():
            if members:
                members[:] = [p for p in members if p not in SYNC_PHASES]
        GROUPS[SYNC_GROUP] = list(SYNC_PHASES)
        GROUP_COLOR[SYNC_GROUP] = SYNC_COLOR
        args.output += "_sync"
    chosen = args.variants.split(",")
    assert set(chosen) <= set(VARIANTS), f"--variants: a subset of {VARIANTS}"
    VARIANTS = [v for v in VARIANTS if v in chosen]
    name = lambda v: label(v, VARIANTS)

    data = {}
    for spec in args.runs:
        m = re.fullmatch(r"(.+)@(\d+)=([^:]+)(?::(.+))?", spec)
        assert m and m[1] in VARIANTS, \
            f"bad run spec {spec!r}, expected VARIANT@RANKS=RUN_DIR[:NOPROF_DIR] with VARIANT in {VARIANTS}"
        data[(int(m[2]), m[1])] = load(m[3], m[4])
    ranks = sorted({r for r, _ in data})
    steps = {d["steps"] for d in data.values()}
    steps_txt = f"{steps.pop()} steps" if len(steps) == 1 else "steps per run differ"

    # groups with no time in any run (e.g. Gravity in wind-shock) are left out of the table and the plot
    active = [g for g in GROUPS if any(d[w][g] > 0 for d in data.values() for w in ("mean", "rank0"))]
    pctg = lambda d: {g: 100 * d[g] / sum(d.values()) for g in active}
    lines = [f"% of step time per phase group ({steps_txt}, first {FIRST_STEP} logged step skipped)",
             "mean = per-rank sums averaged over all ranks (profile.h5); rank0 = rank 0 only;"
             " max = max over ranks per group", ""]
    for g, members in GROUPS.items():
        if g in active:
            lines.append(f"  {g}: {', '.join(members) if members else 'all other phases'}")
    hdr = f"{'ranks':>5}  {'variant':<12}" + "".join(f"{g:>40}" for g in active)
    for which in ("mean", "rank0"):
        lines += ["", f"[{which}]", hdr]
        for r in ranks:
            for v in VARIANTS:
                if (r, v) in data:
                    p = pctg(data[(r, v)][which])
                    lines.append(f"{r:>5}  {name(v):<12}" + "".join(f"{p[g]:39.1f}%" for g in active))
    lines += ["", "seconds per group over the counted steps: rank0 / mean / max over ranks; runtime = max over ranks of",
              "the per-rank total (rank 0's in brackets)", f"{'ranks':>5}  {'variant':<12}{'runtime':>18}"
              + "".join(f"{g:>40}" for g in active)]
    for r in ranks:
        for v in VARIANTS:
            if (r, v) in data:
                d = data[(r, v)]
                lines.append(f"{r:>5}  {name(v):<12}{d['runtime']:9.0f} ({d['runtime_rank0']:6.0f})" + "".join(
                    f"{d['rank0'][g]:>22,.0f} /{d['mean'][g]:>7,.0f} /{d['max'][g]:>7,.0f}" for g in active))
    text = "\n".join(lines) + "\n"
    open(f"{args.output}.txt", "w").write(text)
    print(text)

    if args.absolute:
        fig = draw_absolute(data, ranks, active, steps_txt, args.title, args.narrow)
        png = f"{args.output}_absolute{'_narrow' if args.narrow else ''}.png"
        fig.savefig(png, dpi=150)
        print(f"wrote {args.output}.txt, {png}")
        return

    absolute = args.runtime_heights
    split = log = None
    cuts = {}
    W_H = (6, 7.2) if args.narrow else (12, 6.2)
    if args.log:
        log = 10 ** math.floor(math.log10(min(d["runtime"] for d in data.values())))
        fig, ax = plt.subplots(figsize=W_H)
        panels = [([ax], VARIANTS)]
    elif absolute and len(VARIANTS) == 3:
        # true heights on linear axes: nonmixd-BT0 is 8-33x taller, so it gets its own graph
        fig, axes = plt.subplots(1, 2, figsize=(7.5, 6.8) if args.narrow else (13, 6.2), gridspec_kw={"width_ratios": [1, 1.6]})
        panels = [([axes[0]], VARIANTS[:1]), ([axes[1]], VARIANTS[1:])]
    else:
        runtimes = lambda vs: [d["runtime"] for (_, v), d in data.items() if v in vs]
        shorter = max(runtimes(VARIANTS[1:]), default=0)
        tall = [k for k, d in data.items() if k[1] == VARIANTS[0] and d["runtime"] > CLIP_RATIO * shorter]
        if absolute and len(VARIANTS) > 1 and tall and not args.split_axis:
            # cut each tall bar inside its largest segment, down to CUT_KEEP x the tallest uncut bar or bar part
            largest = {k: max(active, key=lambda g: data[k]["mean"][g]) for k in tall}
            rest = lambda k: data[k]["runtime"] * (1 - pctg(data[k]["mean"])[largest[k]] / 100)
            ref = max([d["runtime"] for k, d in data.items() if k not in tall] + [rest(k) for k in tall])
            for k in tall:
                seg = data[k]["runtime"] * pctg(data[k]["mean"])[largest[k]] / 100
                if seg > CUT_KEEP * ref:
                    cuts[k] = (largest[k], seg - CUT_KEEP * ref)
            fig, ax = plt.subplots(figsize=W_H)
            panels = [([ax], VARIANTS)]
        elif absolute and len(VARIANTS) > 1 and tall:
            split = SPLIT_HEADROOM * shorter
            fig, (up, lo) = plt.subplots(2, 1, sharex=True, figsize=(6, 8.8) if args.narrow else (12, 7.2),
                                         gridspec_kw={"height_ratios": STYLE["split_heights"]})
            panels = [([lo, up], VARIANTS)]
        else:
            fig, ax = plt.subplots(figsize=W_H)
            panels = [([ax], VARIANTS)]
    for axs, variants in panels:
        draw(axs, data, ranks, variants, pctg, absolute, steps_txt, split, log, cuts)
    if absolute and len(VARIANTS) == 3 and not log:
        note = ("Speedup = reference runtime / runtime (> 1: faster). Bracket: mixd-BT0 vs BT3; "
                "row under each group: BT3 and mixd-BT0 vs nonmixd-BT0")
    elif absolute and len(VARIANTS) == 2:
        ref, case = map(name, VARIANTS)
        note = (f"Bracket: speedup = {ref} runtime / {case} runtime (> 1: faster)"
                + ("; y-axis split, upper part on a coarser scale" if split else "")
                + ("; tall bars are cut at // (time taken out written above the cut), the y-axis holds below the cuts"
                   if cuts else "")
                + ("; log y-axis" if log else ""))
    else:
        note = None
    handles = panels[0][0][0].get_legend_handles_labels()
    if args.narrow:  # header stacked top-down at fixed distances in inches, wrapped to the figure width
        h_in = fig.get_figheight()
        y = lambda inch: 1 - inch / h_in
        title = textwrap.fill(f"{args.title} ({steps_txt})", 48)
        suptitle = fig.suptitle(title, y=y(0.08), va="top", fontsize=13, color=TEXT)
        top = 0.12 + 0.25 * (title.count("\n") + 1)
        fig.legend(*handles, loc="upper center", bbox_to_anchor=(0.5, y(top)), ncol=1, frameon=False, fontsize=11,
                   labelcolor=TEXT, handlelength=1.2, borderaxespad=0, labelspacing=0.25)
        top += 0.2 * len(active) + 0.1
        if note:  # also carries what the shortened y label leaves out
            note += ("; 1 GPU per rank. Segments: share of step time (mean over ranks)"
                     + (", as a share of the drawn bar (only bar tops read on the axis)" if log else ""))
            note = textwrap.fill(note, 64)
            fig.text(0.5, y(top), note, ha="center", va="top", fontsize=9.5, color=TEXT2)
            top += 0.165 * (note.count("\n") + 1) + 0.05
        # fit the graph right under the header as drawn (the estimate above only places the header items)
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        bottom = min(a.get_window_extent(r).y0 for a in [suptitle, *fig.texts, *fig.legends]) / fig.bbox.height
        suptitle.set_in_layout(False)  # already above `bottom`; tight_layout would reserve its height again
        fig.tight_layout(rect=(0, 0.01, 1, bottom - 0.01), pad=0.4, h_pad=0.6 if split else None)
    else:
        fig.suptitle(f"{args.title} ({steps_txt})", fontsize=12, color=TEXT)
        fig.legend(*handles, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=len(active), frameon=False,
                   fontsize=9.5, labelcolor=TEXT)
        if note:
            fig.text(0.5, 0.885, note, ha="center", fontsize=8.5, color=TEXT2)
        fig.tight_layout(rect=(0, 0.02, 1, 0.87 if absolute else 0.89), h_pad=0.6 if split else None)
    png = f"{args.output}_runtime{'_log' if log else ''}{'_narrow' if args.narrow else ''}.png" if absolute else f"{args.output}.png"
    fig.savefig(png, dpi=150)
    print(f"wrote {args.output}.txt, {png}")


if __name__ == "__main__":
    main()
