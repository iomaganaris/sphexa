#!/usr/bin/env python3
"""
Per-rank particle distribution from coords_rank*.txt, sampling random lines instead of parsing every file.

Usage: python plot_rank_distribution_sampled.py DIR [--subsample N] [--tag TAG] [--label LABEL] [--section-y W]

Prints per-rank particle and halo counts (first step of DIR/profile.h5) and the 5-95% extents of the sampled
particles, then writes TAG_<N/1000>k_<nranks>ranks.png/.html with the plotting functions of plot_rank_distribution.py,
with larger, more opaque dots than its defaults (fewer samples, each dot's rank colour readable). --section-y W adds
an x-z cross section of the sampled particles within W of the middle y of the computational domain (first
"Computational domain:" line of DIR/run.log, else of the sampled box) below the 3D plot (PNG only).
"""

import argparse
import glob
import os
import re
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_rank_distribution import plot_static, plot_interactive


def sample_file(fn, n, rng):
    """Read n random lines by seeking to random byte offsets and skipping to the next line start."""
    size = os.path.getsize(fn)
    offsets = np.sort(rng.integers(0, max(size - 200, 1), n))
    out = np.empty((n, 3))
    with open(fn, "rb") as f:
        for i, o in enumerate(offsets):
            f.seek(o)
            f.readline()
            out[i] = [float(v) for v in f.readline().split()]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir")
    parser.add_argument("--subsample", type=int, default=5000, help="particles sampled per rank")
    parser.add_argument("--tag", default=None, help="output file prefix (default: basename of DIR)")
    parser.add_argument("--label", default=None, help="run name shown in the plot title, e.g. \"Hybrid-Dimensional SFC\"")
    parser.add_argument("--section-y", type=float, default=None,
                        help="half width of an x-z cross section at the domain's middle y, drawn below the PNG's 3D plot")
    args = parser.parse_args()

    d = args.dir.rstrip("/")
    tag = args.tag or os.path.basename(d)
    files = sorted(glob.glob(f"{d}/coords_rank*.txt"), key=lambda f: int(re.search(r"rank(\d+)\.txt", f).group(1)))
    nranks = len(files)
    if nranks == 0:
        sys.exit(f"No coords_rank*.txt files found in {d}")

    rng = np.random.default_rng(42)
    pos = [sample_file(f, args.subsample, rng) for f in files]
    P = np.vstack(pos)
    ranks = np.concatenate([np.full(args.subsample, r, dtype=np.int32) for r in range(nranks)])
    box = np.array([P[:, 0].min(), P[:, 0].max(), P[:, 1].min(), P[:, 1].max(), P[:, 2].min(), P[:, 2].max()])

    # a profile file can hold several dumps, keep the longest; data is rank-major
    ds = {}
    with h5py.File(f"{d}/profile.h5") as f:
        f.visititems(lambda k, o: ds.__setitem__(k.split("/")[-1], o[:])
                     if isinstance(o, h5py.Dataset) and o.size >= ds.get(k.split("/")[-1], np.zeros(0)).size else None)
    npart = ds["numParticles"].reshape(nranks, -1)[:, 0]
    halos = ds["numHalos"].reshape(nranks, -1)[:, 0]

    print(f"== {tag}: {nranks} ranks, particles/rank {npart.min():,}-{npart.max():,} "
          f"(max/mean {npart.max() / npart.mean():.3f}), halos/rank {halos.min():,}-{halos.max():,} (sum {halos.sum():.3e})")
    print(f"{'rank':>4} {'particles':>10} {'halos':>9} {'halo/part':>9} "
          f"{'x5%':>7} {'x95%':>7} {'y5%':>7} {'y95%':>7} {'z5%':>6} {'z95%':>6}")
    for r in range(nranks):
        q = np.percentile(pos[r], [5, 95], axis=0)
        print(f"{r:4d} {npart[r]:10,} {halos[r]:9,} {halos[r] / npart[r]:9.3f} "
              f"{q[0, 0]:7.2f} {q[1, 0]:7.2f} {q[0, 1]:7.2f} {q[1, 1]:7.2f} {q[0, 2]:6.2f} {q[1, 2]:6.2f}")

    # cut at the middle of the run's computational domain, the same for all runs of a setup (the sampled box depends on
    # which outlying particles the sample happens to pick)
    section_at = None
    if args.section_y is not None:
        log = f"{d}/run.log"
        lines = [l for l in open(log) if "Computational domain:" in l] if os.path.exists(log) else []
        if lines:
            dom = [float(v) for v in lines[0].split("Computational domain:")[1].split()]
            section_at = (dom[2] + dom[3]) / 2
            print(f"cross section at y = {section_at:.3f} (middle of the computational domain in {log})")
        else:
            print(f"WARNING: no 'Computational domain:' line in {log}, cutting at the middle of the sampled box")

    out = f"{tag}_{args.subsample / 1000:g}k_{nranks}ranks"
    plot_static(P, ranks, box, out + ".png", nranks, n_total=int(npart.sum()), label=args.label, size=2, alpha=0.8,
                section_y=args.section_y, section_at=section_at)
    plot_interactive(P, ranks, box, out + ".html", nranks, n_total=int(npart.sum()), label=args.label, size=2.5,
                     opacity=0.8)


if __name__ == "__main__":
    main()
