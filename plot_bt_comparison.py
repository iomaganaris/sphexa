#!/usr/bin/env python3
"""Total runtime and halo comparison of nonmixd-BT0, BT3 and mixd-BT0 from a *_results.txt summary.

  * total runtime [s] = mean 'Total time for iteration' (rank 0) x number of steps counted in the results file
  * total halos = mean over the counted steps of the halos summed over all ranks
  * optional (--busiest): halos of the busiest rank (log 'Halos')

Brackets mark BT3 vs nonmixd-BT0, mixd-BT0 vs BT3 and mixd-BT0 vs nonmixd-BT0 (percent change w.r.t. the
second-named case).

--variants picks and renames columns of the results table (SRC:NAME,...), e.g. for runs labelled differently:
  --variants BT0_nonMixD:nonmixd-BT0,BT3:BT3,BT0:mixd-BT0
"""

import argparse
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

VARIANTS = ["nonmixd-BT0", "BT3", "mixd-BT0"]
COLORS = {"nonmixd-BT0": "#2a78d6", "BT3": "#eb6834", "mixd-BT0": "#1baf7a"}
# (case, reference): percent change of case w.r.t. reference
PAIRS = [("BT3", "nonmixd-BT0"), ("mixd-BT0", "BT3"), ("mixd-BT0", "nonmixd-BT0")]
# build colours of the halo and --absolute plots: hues unused by the phase groups, far apart in greyscale
# (print grey 110 / 112 / 179: nonmixd and BT3 only meet in 3-variant plots); nonmixd is a mid-tone so the near-black
# min..max whisker of the spread plots shows over it; the light gold bars get a dark outline (HALO_EDGE) to stand
# out from the white page
HALO_COLORS = {"nonmixd-BT0": "#4a7fb5", "BT3": "#c2408f", "mixd-BT0": "#e8b425"}
HALO_EDGE = {"mixd-BT0": "#52514e"}
TEXT, TEXT2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
CASE_NAMES = {"windshock": "Wind-shock", "disk": "Protoplanetary Disk"}
PARTICLES = 32301610  # 'Data generated for ... global particles' in windshock_r16_c72_s200_n200_*/run.log


def parse(path, variants):
    lines = open(path).read().splitlines()
    columns = next(l.split()[1:] for l in lines if l.startswith("phase "))
    data = {c: {} for c in columns}
    run = None
    section = None
    for line in lines:
        s = line.strip()
        if not s:
            continue
        first = s.split()[0]
        if first in data:
            run = first
        m = re.search(r"iterations \d+\.\.\d+ \((\d+) steps\)", s)
        if m and run:
            data[run]["steps"] = int(m.group(1))
        if s.startswith("halos,"):
            section = "halos"
        elif s.startswith("end of run"):
            section = None
        if s.startswith("Total time for iteration"):
            vals = [float(x) for x in s.split()[-len(columns):]]
            for v, x in zip(columns, vals):
                data[v]["ms_per_step"] = x
        if section == "halos" and first in data:
            def num(pat):  # None when the run has no profile.h5 (e.g. it hit the time limit)
                m = re.search(pat, s)
                return int(m.group(1).replace(",", "")) if m else None
            data[first]["halos_per_rank"] = num(r"([\d,]+) per rank")
            data[first]["halos_total"] = num(r"([\d,]+) total over ranks")
            data[first]["halos_busiest"] = num(r"busiest rank \(log 'Halos'\)\s+([\d,]+)")
    out = {}
    for src, name in variants:
        d = data[src]
        d["runtime_s"] = d["ms_per_step"] * d["steps"] / 1000.0
        out[name] = d
    return out


def pct(case, ref):
    return 100.0 * (case - ref) / ref


def display_name(variant, variants):
    """Label shown in the figures: the -BT0 suffix is dropped when no BT3 run is shown next to it."""
    return variant if "BT3" in variants else variant.removesuffix("-BT0")


def panel(ax, names, values, title, ylabel):
    x = range(len(names))
    ymax = max(v for v in values if v is not None)
    ax.bar(x, [v or 0 for v in values], width=0.6, color=[COLORS[v] for v in names], edgecolor="white",
           linewidth=2, zorder=2)
    for i, v in enumerate(values):
        if v is None:
            ax.text(i, 0.02 * ymax, "n/a\n(no profile.h5)", ha="center", va="bottom", fontsize=8.5, color=TEXT2)

    idx = {v: i for i, v in enumerate(names) if values[i] is not None}
    pairs = [(c, r) for c, r in PAIRS if c in idx and r in idx]
    for level, (case, ref) in enumerate(pairs):
        a, b = sorted((idx[case], idx[ref]))
        y = ymax * (1.08 + 0.15 * level)
        h = 0.025 * ymax
        ax.plot([a, a, b, b], [y - h, y, y, y - h], color=TEXT2, lw=1)
        d = pct(values[idx[case]], values[idx[ref]])
        ax.text((a + b) / 2, y + 0.01 * ymax, f"{case} vs {ref}: {d:+.1f}%",
                ha="center", va="bottom", fontsize=8.5, color=TEXT)

    ax.set_ylim(0, ymax * (1.05 + 0.15 * len(pairs)))
    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    ax.set_title(title, fontsize=11, color=TEXT)
    ax.set_ylabel(ylabel, color=TEXT2)
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(TEXT2)
    ax.tick_params(colors=TEXT2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="?", default="windshock_r16_s200_n200_results.txt")
    ap.add_argument("-o", "--output", help="output PNG (default: <results stem>_runtime.png)")
    ap.add_argument("--busiest", action="store_true", help="add a panel with the busiest-rank halos")
    ap.add_argument("--particles", type=int, default=PARTICLES, help="global particle count for the title")
    ap.add_argument("--variants", default=",".join(VARIANTS),
                    help=f"SRC[:NAME],... columns to plot, 2 or 3 NAMEs from {VARIANTS} (default: %(default)s)")
    args = ap.parse_args()

    variants = [tuple((v + ":" + v).split(":")[:2]) for v in args.variants.split(",")]
    chosen = [n for _, n in variants]
    assert len(set(chosen)) == len(chosen) >= 2 and set(chosen) <= set(VARIANTS), \
        f"--variants must name 2 or 3 distinct entries of {VARIANTS}"
    names = [v for v in VARIANTS if v in chosen]
    data = parse(args.results, variants)
    stem = re.sub(r"_results$", "", os.path.splitext(os.path.basename(args.results))[0])
    out = args.output or f"{stem}_runtime.png"

    steps = {data[v]["steps"] for v in names}
    steps_txt = f"{steps.pop()} steps" if len(steps) == 1 else "steps per run differ"
    case = stem.split("_")[0]
    ranks = int(re.search(r"_r(\d+)", stem).group(1))
    title = (f"{CASE_NAMES.get(case, case.capitalize())} experiment comparison "
             f"({ranks} ranks, {ranks} GPUs, {steps_txt}, {args.particles / 1e6:.1f}M particles)")

    ncols = 3 if args.busiest else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5.2 * ncols, 5.2))
    panel(axes[0], names, [data[v]["runtime_s"] for v in names], "Runtime comparison", "Runtime [s]")
    mega = lambda v: None if v is None else v / 1e6
    panel(axes[1], names, [mega(data[v]["halos_total"]) for v in names], "Halo comparison", "Halos [millions]")
    if args.busiest:
        panel(axes[2], names, [data[v]["halos_busiest"] / 1e3 for v in names],
              "Busiest-rank halo comparison", "Halos [thousands]")
    fig.suptitle(title, fontsize=12, color=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=150)
    print(f"wrote {out}\n")

    vs = f"vs {names[0]}"
    print(f"{'':14}{'runtime [s]':>12}{vs:>17}{'halos total':>14}{vs:>17}{'busiest':>11}{vs:>17}")
    b = data[names[0]]
    num = lambda v, w: f"{v:>{w},}" if v is not None else f"{'n/a':>{w}}"
    rel = lambda v, r, w: f"{pct(v, r):+{w}.1f}%" if None not in (v, r) else f"{'n/a':>{w + 1}}"
    for v in names:
        d = data[v]
        print(f"{v:14}{d['runtime_s']:12.3f}{rel(d['runtime_s'], b['runtime_s'], 16)}"
              f"{num(d['halos_total'], 14)}{rel(d['halos_total'], b['halos_total'], 16)}"
              f"{num(d['halos_busiest'], 11)}{rel(d['halos_busiest'], b['halos_busiest'], 16)}")
    print()
    for case, ref in [(c, r) for c, r in PAIRS if c in names and r in names]:
        print(f"{case:>9} vs {ref:<12} runtime {rel(data[case]['runtime_s'], data[ref]['runtime_s'], 6)}   "
              f"halos total {rel(data[case]['halos_total'], data[ref]['halos_total'], 6)}")

if __name__ == "__main__":
    main()
