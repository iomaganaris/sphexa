#!/usr/bin/env python3
"""Runtime and total-halo comparison of nonmixd-BT0, BT3 and mixd-BT0 across rank counts, from several
compare_runs_log.py *_results.txt files (one per rank count, rank count read from '_r<ranks>' in the name).

  * bars grouped by rank count; total runtime [s] = mean 'Total time for iteration' (rank 0) x steps counted
  * brackets per group (bottom to top): BT3 vs nonmixd-BT0, mixd-BT0 vs BT3, mixd-BT0 vs nonmixd-BT0;
    runtime as speedup factor (reference time / case time), halos as percent change w.r.t. the reference
  * a graph whose nonmixd-BT0 bars exceed CLIP_RATIO x the tallest other bar is clipped at
    CLIP_LEVEL x that bar: the clipped bars get a '//' mark and their true value written inside
  * --busiest adds a graph with the busiest-rank halos (log 'Halos')
  * --halos-only drops the runtime graph: total halos and busiest-rank halos only
  * --variants nonmixd-BT0,mixd-BT0 plots a subset (brackets only for the pairs left); without BT3 the
    (BT0) is dropped from every label
  * labels spell the builds 3D SFC / Hybrid-Dimensional SFC
  * --halos-only colours the builds with HALO_COLORS (print-safe, apart from the phase-group colours); the other
    graphs keep COLORS
  * --narrow writes OUT_narrow.png instead: graphs stacked in one 6 in wide column, larger fonts, header wrapped
  * --log-halos draws the total-halo graph on a log axis (floor: the power of 10 below the smallest bar), without
    clipping; bars and brackets are placed on log10(value / floor), the ticks are labelled in the graph's units

Columns named BT0_nonMixD / BT0 / BT3 (older results files) are mapped to nonmixd-BT0 / mixd-BT0 / BT3.
"""

import argparse
import math
import re
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_bt_comparison import COLORS, GRID, HALO_COLORS, HALO_EDGE, PAIRS, TEXT, TEXT2, VARIANTS, display_name, parse, pct

ALIASES = {"nonmixd-BT0": "BT0_nonMixD", "BT3": "BT3", "mixd-BT0": "BT0"}
LABELS = {"nonmixd": "3D SFC", "mixd": "Hybrid-Dimensional SFC"}  # spelling of the build names in the figures
CLIP_RATIO = 3.0
CLIP_LEVEL = 1.25
WIDTH = 0.26
FONT = 1.0  # font scale of the text in the graphs, raised by --narrow
PALETTE, EDGE = COLORS, {}  # bar colour and outline per build, HALO_COLORS / HALO_EDGE with --halos-only


def label(variant, variants):
    """display_name with the build spelled out and the boundary type in brackets, e.g. nonmixd-BT0 -> 3D SFC (BT0),
    or 3D SFC when no BT3 run is shown."""
    build, sep, rest = display_name(variant, variants).partition("-")
    return LABELS.get(build, build) + (f" ({rest})" if sep else "")


def load(path):
    lines = open(path).read().splitlines()
    columns = next(l.split()[1:] for l in lines if l.startswith("phase "))
    variants = [(v if v in columns else ALIASES[v], v) for v in VARIANTS]
    ranks = int(re.search(r"_r(\d+)", path).group(1))
    return ranks, parse(path, variants)


def panel(ax, ranks, values, title, ylabel, fmt_value, fmt_bracket, log=False):
    """values[variant][group]; fmt_bracket(case_value, ref_value) -> label.

    log: bars of height log10(value / floor) on a linear axis with ticks labelled in values (no clipping)."""
    others = max(v for k in VARIANTS if k != VARIANTS[0] for v in values[k])
    clip = floor = None
    if log:
        floor = 10 ** math.floor(math.log10(min(v for k in VARIANTS for v in values[k])))
        drawn = {k: [math.log10(v / floor) for v in values[k]] for k in VARIANTS}
    elif max(values[VARIANTS[0]]) > CLIP_RATIO * others:
        clip = CLIP_LEVEL * others
    if not log:
        drawn = {k: [min(v, clip) if clip else v for v in values[k]] for k in VARIANTS}
    top = max(max(drawn[k]) for k in VARIANTS)
    step = 0.09 * top

    name = lambda k: label(k, VARIANTS)
    offset = lambda j: (j - (len(VARIANTS) - 1) / 2) * WIDTH
    for j, k in enumerate(VARIANTS):
        xs = [g + offset(j) for g in range(len(ranks))]
        ax.bar(xs, drawn[k], width=WIDTH, color=PALETTE[k], edgecolor=EDGE.get(k, "white"), linewidth=1.5, zorder=2,
               label=name(k))
        for x, v, d in zip(xs, values[k], drawn[k]):
            if clip and v > clip:
                yb = 0.80 * clip
                for dy in (0, 0.035 * top):
                    ax.plot([x - WIDTH / 2, x + WIDTH / 2], [yb + dy - 0.015 * top, yb + dy + 0.015 * top],
                            color="white", lw=2.5, zorder=3, solid_capstyle="butt")
                ax.text(x, 0.74 * clip, fmt_value(v), rotation=90, ha="center", va="top", fontsize=8.5 * FONT,
                        color="white", zorder=4)

    idx = {k: j for j, k in enumerate(VARIANTS)}
    for g in range(len(ranks)):
        base = max(drawn[k][g] for k in VARIANTS) + 0.04 * top
        for level, (case, ref) in enumerate(PAIRS):
            a, b = sorted((idx[case], idx[ref]))
            xa, xb = g + offset(a), g + offset(b)
            y = base + level * step
            h = 0.02 * top
            ax.plot([xa, xa, xb, xb], [y - h, y, y, y - h], color=TEXT2, lw=1, zorder=2)
            ax.text((xa + xb) / 2, y + 0.008 * top, fmt_bracket(values[case][g], values[ref][g]),
                    ha="center", va="bottom", fontsize=8 * FONT, color=TEXT)

    ax.set_ylim(0, top * 1.04 + len(PAIRS) * step + 0.06 * top)
    ax.set_xticks(range(len(ranks)))
    ax.set_xticklabels([str(r) for r in ranks])
    ax.set_xlabel("Ranks (GPUs)", color=TEXT2, fontsize=10 * FONT)
    ax.set_title(title, fontsize=11 * FONT, color=TEXT)
    ax.set_ylabel(ylabel + (f" ({name(VARIANTS[0])} clipped)" if clip else "") + (" (log scale)" if log else ""),
                  color=TEXT2, fontsize=10 * FONT)
    if log:
        ticks = [m * 10 ** e for e in range(round(math.log10(floor)), round(math.log10(floor)) + 8) for m in (1, 2, 5)]
        ticks = [t for t in ticks if math.log10(t / floor) <= ax.get_ylim()[1]]
        ax.set_yticks([math.log10(t / floor) for t in ticks])
        ax.set_yticklabels([f"{t:,g}" for t in ticks])
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(TEXT2)
    ax.tick_params(colors=TEXT2, labelsize=10 * FONT)
    return clip


def main():
    global VARIANTS, PAIRS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+", help="*_results.txt files, one per rank count")
    ap.add_argument("-o", "--output", default="bt_scaling.png")
    ap.add_argument("--case", default="Protoplanetary Disk", help="experiment name for the title (default: %(default)s)")
    ap.add_argument("--particles", type=int, help="global particle count for the title")
    ap.add_argument("--busiest", action="store_true", help="add a graph with the busiest-rank halos")
    ap.add_argument("--halos-only", action="store_true",
                    help="only the total-halo and busiest-rank halo graphs (no runtime)")
    ap.add_argument("--variants", default=",".join(VARIANTS),
                    help="comma-separated subset of %(default)s to plot (default: all)")
    ap.add_argument("--log-halos", action="store_true", help="total-halo graph on a log axis, without clipping")
    ap.add_argument("--narrow", action="store_true",
                    help="graphs stacked in one half-page-wide column with larger fonts (writes OUT_narrow.png)")
    args = ap.parse_args()
    global FONT, PALETTE, EDGE
    if args.halos_only:
        PALETTE, EDGE = HALO_COLORS, HALO_EDGE
    if args.narrow:
        FONT = 1.3
        args.output = re.sub(r"(\.png)?$", "_narrow.png", args.output, count=1)

    chosen = args.variants.split(",")
    assert len(chosen) >= 2 and set(chosen) <= set(VARIANTS), f"--variants: 2 or 3 of {VARIANTS}"
    VARIANTS = [v for v in VARIANTS if v in chosen]
    PAIRS = [p for p in PAIRS if set(p) <= set(VARIANTS)]
    name = lambda k: label(k, VARIANTS)

    runs = sorted(load(p) for p in args.results)
    ranks = [r for r, _ in runs]
    get = lambda key, scale=1.0: {k: [d[k][key] / scale for _, d in runs] for k in VARIANTS}

    steps = {d[k]["steps"] for _, d in runs for k in VARIANTS}
    info = [f"{steps.pop()} steps" if len(steps) == 1 else "steps per run differ", "1 GPU per rank"]
    if args.particles:
        info.insert(0, f"{args.particles / 1e6:.1f}M particles")

    speedup = lambda case, ref: f"{ref / case:.2f}×"
    change = lambda case, ref: f"{pct(case, ref):+.1f}%"
    graphs = [] if args.halos_only else [
        (get("runtime_s"), "Runtime comparison", "Runtime [s]", lambda v: f"{v:,.0f} s", speedup)]
    graphs.append((get("halos_total", 1e6), "Halo comparison", "Halos [millions]", lambda v: f"{v:,.0f}M", change,
                   args.log_halos))
    if args.busiest or args.halos_only:
        graphs.append((get("halos_busiest", 1e6), "Busiest-rank halo comparison", "Halos [millions]",
                       lambda v: f"{v:,.1f}M", change))
    if args.narrow:
        fig, axes = plt.subplots(len(graphs), 1, figsize=(6, 1.6 + 3.2 * len(graphs)))
    else:
        fig, axes = plt.subplots(1, len(graphs), figsize=(7.2 * len(graphs), 6.0))
    for ax, g in zip(axes, graphs):
        panel(ax, ranks, *g)

    what = "halo comparison" if args.halos_only else "comparison"
    title = f"{args.case} experiment {what} ({', '.join(info)})"
    handles, labels = axes[0].get_legend_handles_labels()
    pairs = ", ".join(f"{name(c)} vs {name(r)}" for c, r in PAIRS)
    brackets = f"Brackets per group, bottom to top: {pairs}" if len(PAIRS) > 1 else f"Bracket per group: {pairs}"
    brackets += " (% change)" if args.halos_only else " (runtime: speedup, halos: % change)"
    if args.narrow:  # header items stacked top-down, each placed under the previous one as drawn
        r = fig.canvas.get_renderer()
        below = lambda a, gap=0.06: (a.get_window_extent(r).y0 - gap * fig.dpi) / fig.bbox.height
        sup = fig.suptitle(textwrap.fill(title, 48), y=1 - 0.08 / fig.get_figheight(), va="top", fontsize=13,
                           color=TEXT)
        fig.canvas.draw()
        leg = fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, below(sup)), ncol=len(VARIANTS),
                         frameon=False, fontsize=11, labelcolor=TEXT, borderaxespad=0)
        fig.canvas.draw()
        note = fig.text(0.5, below(leg), textwrap.fill(brackets, 64), ha="center", va="top", fontsize=9.5, color=TEXT2)
        fig.canvas.draw()
        for a in (sup, leg, note):
            a.set_in_layout(False)  # already above the rect top; tight_layout would reserve their height again
        fig.tight_layout(rect=(0, 0.01, 1, below(note, 0.04)), pad=0.4, h_pad=1.2)
    else:
        fig.suptitle(title, fontsize=12, color=TEXT)
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=len(VARIANTS),
                   frameon=False, fontsize=9.5, labelcolor=TEXT)
        fig.text(0.5, 0.865, brackets, ha="center", fontsize=8.5, color=TEXT2)
        fig.tight_layout(rect=(0, 0, 1, 0.85))
    fig.savefig(args.output, dpi=150)
    print(f"wrote {args.output}\n")

    print(f"{'ranks':>5} {'variant':<12}{'runtime [s]':>12}{'halos total':>15}{'busiest':>13}")
    for r, d in runs:
        for k in VARIANTS:
            print(f"{r:>5} {name(k):<12}{d[k]['runtime_s']:12.1f}{d[k]['halos_total']:15,}{d[k]['halos_busiest']:13,}")
    print()
    for r, d in runs:
        print(f"{r:>5} " + "   ".join(
            f"{name(c)} vs {name(f)}: {d[f]['runtime_s'] / d[c]['runtime_s']:.2f}x, halos {pct(d[c]['halos_total'], d[f]['halos_total']):+.1f}%"
            for c, f in PAIRS))


if __name__ == "__main__":
    main()
