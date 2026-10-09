#!/usr/bin/env python3
"""Compare sphexa runs from slurm logs: per-phase time (rank 0, as logged) skipping the first FIRST_STEP logged steps,
halos averaged over all ranks and steps (profile.h5 numHalos, rank-major [rank][iteration 1..N]),
the log's Halos (= busiest rank per step) and the final total energy / time step.

Usage: compare_runs_log.py [--common-window] LABEL=SLURM_OUT:RUN_INDEX:RUN_DIR [...]

--common-window: cut every run to the last iteration reached by all runs (e.g. when one run hit the time
limit), so all runs are averaged over the same iterations. 'whole run' is then still the full run's time.
"""
import re, sys, os
import numpy as np
import h5py

FIRST_STEP = 1  # counted from the run's first logged iteration (0-based), not the absolute iteration


def parse_run(path, index):
    runs, cur = [], None
    lines = [l.rstrip("\n") for l in open(path)]
    is_hdr = lambda l: l.startswith("Running ") and "SPHEXA_BOUNDARY_TYPE=" in l
    # logs without the "Running ... SPHEXA_BOUNDARY_TYPE=" header (older scripts): split on the
    # "# SPHEXA: branch/commit" banner that sphexa prints at the start of every run instead
    has_hdr = any(map(is_hdr, lines))
    for l in lines:
        if is_hdr(l) if has_hdr else l.startswith("# SPHEXA: "):
            hdr = l if has_hdr else f"Running run {len(runs)} of {os.path.basename(path)} with {l[2:]}"
            cur = {"hdr": hdr, "its": [], "blk": []}; runs.append(cur)
        elif l.startswith("===== Last"):
            break
        elif cur is None:
            continue
        elif l.startswith("# ") and l.endswith("s") and ": " in l and "Total execution" not in l:
            n, v = l[2:].rsplit(": ", 1)
            try: cur["blk"].append((n, float(v[:-1])))
            except ValueError: pass
        elif "### Check ### Global Tree Nodes" in l:
            cur["h"] = int(l.split("Halos:")[1])
        elif l.startswith("### Check ### Total time:"):
            m = re.search(r"Total time: ([\d.e+-]+), current time-step: ([\d.e+-]+)", l); cur["t"], cur["dt"] = map(float, m.groups())
        elif l.startswith("### Check ### Total energy:"):
            cur["E"] = l.split("Total energy: ")[1]
        elif (m := re.match(r"=== Total time for iteration\((\d+)\) ([\d.e+-]+)s", l)):
            cur["its"].append((int(m[1]), float(m[2]), cur["blk"], cur["h"], cur["t"], cur["dt"], cur["E"])); cur["blk"] = []
        elif l.startswith("# Total execution"):
            cur["exec"] = l.rsplit(": ", 1)[1]
    if index >= len(runs):
        sys.exit(f"{path}: run index {index} requested, but only {len(runs)} run(s) found")
    return runs[index]


def phase_names(blk):
    names = [n for n, _ in blk]
    out, seen = [], {}
    for n in names:
        seen[n] = seen.get(n, 0) + 1
        out.append(n if names.count(n) == 1 else f"{n} ({seen[n]})")
    return out


def profile_halos(run_dir, nsteps, trim=0):
    p = os.path.join(run_dir, "profile.h5")
    if not os.path.exists(p):
        return None
    with h5py.File(p, "r") as f:
        cands = [f[g]["numHalos"] for g in f if "numHalos" in f[g]]
        ds = max(cands, key=lambda d: d.shape[0])
        a = ds.parent.attrs
        nr, ni = int(a["numRanks"][0]), int(a["numIterations"][0])
        h = ds[:].reshape(nr, ni).astype(np.int64)    # [rank][step]
    # step 0 is sometimes stored in a group of its own, so align on the last steps;
    # trim = logged iterations dropped from the end by --common-window
    return h[:, ni - nsteps - trim : ni - trim]


def main(args):
    common = "--common-window" in args
    specs = [a for a in args if a != "--common-window"]
    runs = {}
    for spec in specs:
        label, rest = spec.split("=", 1)
        out, idx, rdir = rest.split(":")
        runs[label] = (parse_run(out, int(idx)), rdir)
    last_it = min(r["its"][-1][0] for r, _ in runs.values()) if common else None
    res = {}
    for label, (r, rdir) in runs.items():
        its = r["its"][FIRST_STEP:]
        trim = 0
        if common:
            trim = sum(1 for i in its if i[0] > last_it)
            its = its[: len(its) - trim]
        names = phase_names(its[0][2])
        A = np.array([[v for _, v in i[2]] for i in its])
        hp = profile_halos(rdir, len(its), trim)
        res[label] = dict(hdr=r["hdr"], names=names, ph=A.mean(0), tot=np.mean([i[1] for i in its]),
                          n=len(its), last=its[-1], hlog=np.mean([i[3] for i in its]), hp=hp,
                          exec=r.get("exec", "(not finished)"))
    labels = list(res)
    firsts = sorted({r["last"][0] - r["n"] + 1 for r in res.values()})
    skipped = "first logged step" if FIRST_STEP == 1 else f"first {FIRST_STEP} logged steps"
    steps_label = f"iterations >= {'/'.join(map(str, firsts))} ({skipped} skipped)"
    print("runs:")
    for k in labels:
        r = res[k]
        print(f"  {k:<12} {r['hdr'].split(' with ')[0].replace('Running ','')}: {r['hdr'].split(' with ')[1]}")
        full = " (full run)" if last_it is not None and "exec" in runs[k][0] else ""
        print(f"  {'':<12} iterations {r['last'][0] - r['n'] + 1}..{r['last'][0]} ({r['n']} steps), whole run {r['exec']}{full}")
    names = res[labels[0]]["names"]
    print(f"\nms per step, rank 0, mean over {steps_label}")
    print(f"{'phase':<32}" + "".join(f"{k:>12}" for k in labels))
    order = np.argsort(-np.max([res[k]["ph"] for k in labels], axis=0))
    for i in order:
        print(f"{names[i]:<32}" + "".join(f"{res[k]['ph'][i]*1e3:12.2f}" for k in labels))
    print(f"{'Total time for iteration':<32}" + "".join(f"{res[k]['tot']*1e3:12.2f}" for k in labels))
    print(f"\nhalos, mean over {steps_label}")
    for k in labels:
        r = res[k]; hp = r["hp"]
        allr = f"{hp.mean():>12,.0f} per rank, {hp.sum(0).mean():>14,.0f} total over ranks" if hp is not None else f"{'n/a (no profile.h5 yet)':>44}"
        print(f"  {k:<12} all ranks: {allr};  busiest rank (log 'Halos') {r['hlog']:>12,.0f}")
    print("\nend of run")
    for k in labels:
        it, _, _, _, t, dt, E = res[k]["last"]
        print(f"  {k:<12} iteration {it:<4} t = {t:<10g} dt = {dt:<12g} E = {E}")


if __name__ == "__main__":
    main(sys.argv[1:])
