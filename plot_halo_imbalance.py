#!/usr/bin/env python3
"""Halo imbalance across ranks, from the per-rank halo counts (numHalos) in profile.h5.

Usage: plot_halo_imbalance.py [--variants A,B] [--title T] -o OUT.png VARIANT@RANKS=RUN_DIR [...]

  * per run: H_r = halos of rank r summed over the logged iterations, skipping the first FIRST_STEP
    (as compare_runs_log.py); imbalance factor = max_r H_r / mean_r H_r (1 = every rank has the same halo count)
  * bars grouped by rank count, one per variant, labelled with the factor and, below it, the busiest rank's
    mean halos per step; a dashed line marks perfect balance (1.0)
  * --variants nonmixd-BT0,mixd-BT0 plots a subset (labels drop -BT0 when no BT3 run is shown)
  * --spread writes OUT_spread.png instead: bar = mean_r H_r, whisker = min_r H_r .. max_r H_r (linear axis),
    spread max_r H_r - min_r H_r above each whisker; one bracket per rank count compares the first variant with
    the last: mean ratio 3D SFC / Hybrid-Dimensional SFC and spread difference 3D SFC - Hybrid-Dimensional SFC
  * --spread --narrow writes OUT_spread_narrow.png: 6 in wide, larger fonts, header wrapped
"""

import argparse
import re
import textwrap

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compare_runs_log import FIRST_STEP
from plot_bt_comparison import GRID, HALO_COLORS, HALO_EDGE, TEXT, TEXT2, VARIANTS
from plot_bt_scaling import label

WIDTH = 0.36
FONT = 1.0  # font scale of the text in the graph, raised by --narrow


def per_rank_halos(run_dir):
    """Returns ([rank] halos summed over the counted steps, counted steps)."""
    with h5py.File(f"{run_dir}/profile.h5", "r") as f:
        ds = max((f[g]["numHalos"] for g in f if "numHalos" in f[g]), key=lambda d: d.shape[0])
        a = ds.parent.attrs
        nr, ni = int(a["numRanks"][0]), int(a["numIterations"][0])
        h = ds[:].reshape(nr, ni).astype(np.int64)[:, FIRST_STEP:]  # [rank][step]
    return h.sum(1), h.shape[1]


def load(run_dir):
    """Returns (imbalance factor, busiest rank's mean halos per step, counted steps)."""
    per_rank, steps = per_rank_halos(run_dir)
    return per_rank.max() / per_rank.mean(), per_rank.max() / steps, steps


def halos(v):
    return f"{v / 1e6:.1f}M" if v >= 1e6 else f"{v / 1e3:.0f}k"


def units(top):
    """(scale, axis unit, value suffix) for halo totals up to top."""
    return (1e9, "billions", "B") if top >= 1e9 else (1e6, "millions", "M")


def amount(v, suffix, sign=False):
    """v in the axis unit, 3 significant figures, e.g. 4.03B; sign=True prefixes + or −."""
    text = f"{abs(v):#.3g}".rstrip(".") if abs(v) < 1000 else f"{abs(v):,.0f}"
    return ("−" if v < 0 else "+" if sign else "") + text + suffix


def draw_spread(args, data, name):
    """data[(ranks, variant)] = ([rank] halo totals, steps); bars of the mean, whiskers min..max."""
    variants = [v for v in VARIANTS if any(k[1] == v for k in data)]
    ranks = sorted({r for r, _ in data})
    stats = {k: (h.mean(), h.min(), h.max()) for k, (h, _) in data.items()}
    scale, unit, suffix = units(max(hi for _, _, hi in stats.values()))
    top = max(hi for _, _, hi in stats.values()) / scale

    if args.narrow:
        fig, ax = plt.subplots(figsize=(6, 6.3))
    else:
        fig, ax = plt.subplots(figsize=(1.6 + 1.7 * len(ranks), 6.0))
    offset = lambda j: (j - (len(variants) - 1) / 2) * WIDTH
    label_h = 0.075 * top * FONT  # room for the factor label above a whisker
    for j, v in enumerate(variants):
        pts = [(g + offset(j), stats[(r, v)]) for g, r in enumerate(ranks) if (r, v) in stats]
        xs = [x for x, _ in pts]
        mean, lo, hi = (np.array([s[i] for _, s in pts]) / scale for i in range(3))
        ax.bar(xs, mean, width=WIDTH, color=HALO_COLORS[v], edgecolor=HALO_EDGE.get(v, "white"), linewidth=1.5,
               zorder=2, label=name(v))
        ax.errorbar(xs, mean, yerr=[mean - lo, hi - mean], fmt="none", ecolor=TEXT, elinewidth=1.3,
                    capsize=5 * FONT, capthick=1.3, zorder=3)
        for x, l, h in zip(xs, lo, hi):
            ax.text(x, h + 0.012 * top, amount(h - l, suffix), ha="center", va="bottom", fontsize=8.5 * FONT, color=TEXT)

    first, last = variants[0], variants[-1]
    brackets = []
    for g, r in enumerate(ranks):
        if (r, first) not in stats or (r, last) not in stats:
            continue
        (m1, lo1, hi1), (m2, lo2, hi2) = stats[(r, first)], stats[(r, last)]
        xa, xb = g + offset(0), g + offset(len(variants) - 1)
        y = max(hi1, hi2) / scale + label_h + 0.03 * top
        h = 0.02 * top
        ax.plot([xa, xa, xb, xb], [y - h, y, y, y - h], color=TEXT2, lw=1, zorder=2)
        ax.text((xa + xb) / 2, y + 0.01 * top, f"mean {m1 / m2:.2f}×\nspread {amount(((hi1 - lo1) - (hi2 - lo2)) / scale, suffix, sign=True)}",
                ha="center", va="bottom", fontsize=8 * FONT, color=TEXT, linespacing=1.15)
        brackets.append(y)

    ax.set_ylim(0, max(brackets, default=top) + 0.17 * top * FONT)
    ax.set_xlim(-0.5, len(ranks) - 0.5)
    ax.set_xticks(range(len(ranks)))
    ax.set_xticklabels([str(r) for r in ranks])
    ax.set_xlabel("Ranks (GPUs)", color=TEXT2, fontsize=10 * FONT)
    ax.set_ylabel(f"Halos per rank over the run [{unit}]", color=TEXT2, fontsize=10 * FONT)
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(TEXT2)
    ax.tick_params(colors=TEXT2, labelsize=10 * FONT)

    steps = {s for _, s in data.values()}
    steps_txt = f"{steps.pop()} steps" if len(steps) == 1 else "steps per run differ"
    note = (f"Bar: mean over ranks of each rank's halos summed over {steps_txt}; whisker: min to max over ranks; "
            f"label: spread = max − min over ranks. Bracket: mean {name(first)} / {name(last)}, "
            f"spread {name(first)} − {name(last)}")
    handles, labels = ax.get_legend_handles_labels()
    if args.narrow:  # header items stacked top-down, each placed under the previous one as drawn
        r = fig.canvas.get_renderer()
        below = lambda a, gap=0.06: (a.get_window_extent(r).y0 - gap * fig.dpi) / fig.bbox.height
        sup = fig.suptitle(textwrap.fill(args.title, 48), y=1 - 0.08 / fig.get_figheight(), va="top", fontsize=13,
                           color=TEXT)
        fig.canvas.draw()
        leg = fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, below(sup)), ncol=len(variants),
                         frameon=False, fontsize=11, labelcolor=TEXT, borderaxespad=0)
        fig.canvas.draw()
        txt = fig.text(0.5, below(leg), textwrap.fill(note, 70), ha="center", va="top", fontsize=9.5, color=TEXT2)
        fig.canvas.draw()
        for a in (sup, leg, txt):
            a.set_in_layout(False)  # already above the rect top; tight_layout would reserve their height again
        fig.tight_layout(rect=(0, 0.01, 1, below(txt, 0.04)), pad=0.4)
    else:
        fig.suptitle(args.title, fontsize=11.5, color=TEXT)
        ax.set_title(textwrap.fill(note, int(14 * fig.get_figwidth())), fontsize=8.5, color=TEXT2)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=len(variants), frameon=False, fontsize=9,
                  labelcolor=TEXT)
        fig.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches=None if args.narrow else "tight")
    print(f"wrote {args.output}\n")

    print(f"{'ranks':>5} {'variant':<10}{'mean':>16}{'min':>16}{'max':>16}{'max-min':>16}{'max/mean':>10}{'steps':>7}")
    for r in ranks:
        for v in variants:
            if (r, v) in stats:
                m, lo, hi = stats[(r, v)]
                print(f"{r:>5} {name(v):<10}{m:16,.0f}{lo:16,}{hi:16,}{hi - lo:16,}{hi / m:10.2f}{data[(r, v)][1]:7}")
    print()
    for r in ranks:
        if (r, first) in stats and (r, last) in stats:
            (m1, lo1, hi1), (m2, lo2, hi2) = stats[(r, first)], stats[(r, last)]
            print(f"{r:>5} {name(first)} / {name(last)}: mean {m1 / m2:.2f}x, spread {(hi1 - lo1) - (hi2 - lo2):+,}")


def main():
    global VARIANTS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="VARIANT@RANKS=RUN_DIR")
    ap.add_argument("-o", "--output", default="halo_imbalance.png")
    ap.add_argument("--title", default="Halo imbalance across ranks (max / mean)")
    ap.add_argument("--variants", default=",".join(VARIANTS),
                    help="comma-separated subset of %(default)s to plot (default: all)")
    ap.add_argument("--spread", action="store_true",
                    help="mean bar with min..max whisker per run, brackets 3D SFC / Hybrid-Dimensional SFC (writes OUT_spread.png)")
    ap.add_argument("--narrow", action="store_true", help="with --spread: 6 in wide, larger fonts (OUT_spread_narrow.png)")
    args = ap.parse_args()
    assert not args.narrow or args.spread, "--narrow needs --spread"
    chosen = args.variants.split(",")
    assert set(chosen) <= set(VARIANTS), f"--variants: a subset of {VARIANTS}"
    VARIANTS = [v for v in VARIANTS if v in chosen]
    name = lambda v: label(v, VARIANTS)

    data = {}
    for spec in args.runs:
        m = re.fullmatch(r"(.+)@(\d+)=(.+)", spec)
        assert m and m[1] in VARIANTS, f"bad run spec {spec!r}, expected VARIANT@RANKS=RUN_DIR with VARIANT in {VARIANTS}"
        data[(int(m[2]), m[1])] = per_rank_halos(m[3]) if args.spread else load(m[3])
    if args.spread:
        global FONT
        FONT = 1.3 if args.narrow else 1.0
        args.output = re.sub(r"(\.png)?$", "_spread_narrow.png" if args.narrow else "_spread.png", args.output, count=1)
        draw_spread(args, data, name)
        return
    ranks = sorted({r for r, _ in data})
    steps = {s for _, _, s in data.values()}

    fig, ax = plt.subplots(figsize=(1.6 + 1.5 * len(ranks), 5.6))
    top = max(f for f, _, _ in data.values())
    offset = lambda j: (j - (len(VARIANTS) - 1) / 2) * WIDTH
    for j, v in enumerate(VARIANTS):
        pts = [(g + offset(j), data[(r, v)]) for g, r in enumerate(ranks) if (r, v) in data]
        ax.bar([x for x, _ in pts], [d[0] for _, d in pts], width=WIDTH, color=HALO_COLORS[v],
               edgecolor=HALO_EDGE.get(v, "white"), linewidth=1.5, zorder=2, label=name(v))
        for x, (factor, busiest, _) in pts:
            ax.text(x, factor + 0.075 * top, f"{factor:.2f}×", ha="center", va="bottom", fontsize=9, color=TEXT)
            ax.text(x, factor + 0.015 * top, halos(busiest), ha="center", va="bottom", fontsize=7.5, color=TEXT2)
    ax.axhline(1.0, color=TEXT2, lw=1, ls="--", zorder=3)
    ax.text(len(ranks) - 0.5, 1.0, " perfect balance", ha="left", va="center", fontsize=8, color=TEXT2,
            clip_on=False)

    ax.set_ylim(0, top * 1.2)
    ax.set_xlim(-0.5, len(ranks) - 0.5)
    ax.set_xticks(range(len(ranks)))
    ax.set_xticklabels([str(r) for r in ranks])
    ax.set_xlabel("Ranks (GPUs)", color=TEXT2)
    ax.set_ylabel("Halo imbalance: max / mean over ranks", color=TEXT2)
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(TEXT2)
    ax.tick_params(colors=TEXT2)

    steps_txt = f"{steps.pop()} steps" if len(steps) == 1 else "steps per run differ"
    fig.suptitle(args.title, fontsize=11.5, color=TEXT)
    ax.set_title(f"Halos per rank summed over {steps_txt}; small label: busiest rank's mean halos per step",
                 fontsize=8.5, color=TEXT2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=len(VARIANTS), frameon=False, fontsize=9,
              labelcolor=TEXT)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"wrote {args.output}\n")

    print(f"{'ranks':>5} {'variant':<10}{'max/mean':>10}{'busiest/step':>15}{'steps':>7}")
    for r in ranks:
        for v in VARIANTS:
            if (r, v) in data:
                factor, busiest, s = data[(r, v)]
                print(f"{r:>5} {name(v):<10}{factor:10.2f}{busiest:15,.0f}{s:7}")


if __name__ == "__main__":
    main()
