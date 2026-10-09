# Handoff: comparison plots (Hybrid-Dimensional SFC vs 3D SFC, BT3 vs BT0)

How to regenerate every comparison plot made on 2026-10-01. All commands run from the repo root
`/capstor/scratch/cscs/ioannmag/CORNERSTONE/sphexa`.

## Setup

```bash
P=$(pwd)/venv_sphexa_new/bin/python3   # matplotlib 3.11, h5py, plotly
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1   # needed when running several plots in parallel
```

Runs must be launched with `--profile -op <dir>/profile` so `<dir>/profile.h5` exists: the per-rank halo totals
and the phase-group plots need it. Rank-distribution plots also need `SPHEXA_WRITE_RANK_COORDS=1`
(writes `coords_rank*.txt`).

## Pipeline

1. `compare_runs_log.py` reads a slurm log (or a run folder's `run.log`) plus `profile.h5` and writes a
   `*_results.txt` table, one per rank count.
2. `plot_bt_scaling.py` reads one or more `*_results.txt` (rank count parsed from `_r<ranks>` in the name)
   and draws runtime and halo bar charts grouped by rank count.
3. `plot_phase_groups.py` reads `profile.h5` directly and draws the time per phase group.
4. `plot_rank_distribution_sampled.py` draws the 3D particle-to-rank scatter (PNG + interactive HTML).
5. `plot_halo_imbalance.py` reads `profile.h5` directly and draws the halo imbalance across ranks (max / mean),
   or with `--spread` the mean with min..max whiskers.

### Variant names

Internal names (used in `--variants`, run specs and results-table columns): `nonmixd-BT0`, `BT3`, `mixd-BT0`.
In figures the builds are spelled **3D SFC / Hybrid-Dimensional SFC** (`LABELS` in `plot_bt_scaling.py`); `(BT0)` is
dropped when BT3 is not in the plot (`nonmixd-BT0,mixd-BT0` -> `3D SFC`, `Hybrid-Dimensional SFC`; with BT3 ->
`3D SFC (BT0)`, `BT3`, `Hybrid-Dimensional SFC (BT0)`). Older spellings stay in plots until they are regenerated:
- Hybrid-Dimensional SFC (since 2026-10-09): only `disk_chain2x4_mixd_vs_nonmixd_phase_groups_runtime_log_narrow.png`,
  `disk_chain2x4_mixd_vs_nonmixd_halo_spread_narrow.png`, `windshock_n400_mixd_vs_nonmixd_halo_spread_narrow.png`,
  `windshock_n400_mixd_vs_nonmixd_phase_groups_runtime_narrow.png`, and the rank-distribution plots
  `windshock_n400_mixd_100k_16ranks.png/.html`, `disk_chain2x4_mixd_100k_16ranks.png/.html` (their 3D SFC
  counterparts are unchanged: same seeded sample, same title).
- Mixed dimensional SFC (2026-10-05 to 10-08): the other phase-group runtime and absolute plots (wind-shock, disk,
  TDE), the other halo-spread / halo-imbalance plots and the wind-shock scaling plots.
- nonMixD / MixD (before 2026-10-05): the disk and TDE scaling plots and the other rank-distribution plots.

- 3D SFC (nonMixD) = `build_gpu_release_nonmixd`, Hybrid-Dimensional SFC (MixD) = `build_gpu_release`; BT = `SPHEXA_BOUNDARY_TYPE`.

### Colours (black-and-white print)

Every colour pair that shares a figure is far apart in print grey (0.299 R + 0.587 G + 0.114 B):
- phase groups (`GROUP_COLORS` in `plot_phase_groups.py`, also used by `plot_stage_timeline.py`): Domain
  decomposition navy `#1c3f74` (59), Particle interaction green `#2e9a68` (116), Gravity light orange `#f4a259`
  (178), Waiting for the slowest rank light grey `#e3e1dc` (225) with a `///` hatch; share text inside a segment is
  white on navy/green, black on orange/grey; cut marks and cut text dark grey `#52514e`
- builds in the halo plots (`_halo_imbalance`, `_halo_spread`, `_scaling_halos[_log]`) and `--absolute`
  (`HALO_COLORS` in `plot_bt_comparison.py`): 3D SFC steel blue `#4a7fb5` (110), BT3 magenta `#c2408f` (112),
  Hybrid-Dimensional SFC gold `#e8b425` (179, dark outline). 3D SFC is a mid-tone so the near-black min..max whisker of
  the spread plots shows over it; it matches BT3 in print grey, but the two only meet in 3-variant plots. Changed
  on 2026-10-05 from dark purple `#3b1f5c` (46); regenerated with it so far: `*_halo_imbalance.png` and
  `*_halo_spread[_narrow].png` (wind-shock, disk), and on 2026-10-08 the wind-shock `_scaling_halos[_narrow]` and
  `--absolute` plots. The disk `_scaling_halos[_log][_narrow]` and `--absolute` plots keep the purple until
  regenerated.
- not print-safe, kept on purpose: `COLORS` (3D SFC blue / BT3 orange / Hybrid-Dimensional SFC green, grey 107 / 137 / 125) in
  `_scaling_busiest[_log]`, `plot_bt_comparison.py`, `plot_stage_shares.py` and the old 3-variant plots
  (`windshock_scaling_halos.png`, `disk_chain2x4_phase_groups*.png`, ...), which were not regenerated

## 1. compare_runs_log.py

```bash
$P compare_runs_log.py [--common-window] LABEL=LOG:RUN_INDEX:RUN_DIR [...] > NAME_r<ranks>_..._results.txt
```

- `LOG` = slurm `.out` or `run.log`; `RUN_INDEX` = which run inside the log (0-based; a log with
  `Running with SPHEXA_BOUNDARY_TYPE=0` then `=3` has BT0 at 0, BT3 at 1); `RUN_DIR` = folder with `profile.h5`.
- The first logged step is skipped (`FIRST_STEP = 1`); reading stops at `===== Last`.
- Halo columns: "total over ranks" = mean over steps of the sum over ranks (profile.h5);
  "busiest rank (log 'Halos')" = the `### Check ### Global Tree Nodes ... Halos:` line, which is the
  **maximum over ranks**, not rank 0 (verified against profile.h5 on the TDE run).
- `--common-window` if one run stopped early (time limit).

## 2. plot_bt_scaling.py

```bash
$P plot_bt_scaling.py [--busiest | --halos-only] [--log-halos] --case NAME --particles N \
    --variants A,B -o OUT.png RESULTS_r*.txt
```

| option | effect |
|---|---|
| (default) | runtime + total-halo graphs |
| `--busiest` | adds busiest-rank halo graph -> the `_scaling_busiest.png` plots |
| `--halos-only` | total + busiest-rank halos, no runtime -> the `_scaling_halos.png` plots |
| `--log-halos` | total-halo graph on a log axis (floor = power of 10 below the smallest bar), no clipping |
| `--variants` | 2 or 3 of `nonmixd-BT0,BT3,mixd-BT0` (default all 3) |
| `--narrow` | writes `OUT_narrow.png`: graphs stacked in one 6 in wide column (1.6 + 3.2 in per graph high), larger fonts, wrapped header |

Runtime brackets show speedup (reference / case), halo brackets show % change. A graph whose first variant is
> 3x the others is clipped (`//` mark, true value written in the bar) unless it is log.

## 3. plot_phase_groups.py

```bash
$P plot_phase_groups.py --runtime-heights [--log] --variants A,B --title "..." -o OUT_STEM \
    VARIANT@RANKS=RUN_DIR [...]
```

- Groups: *Domain decomposition and communication* (domain::sync, computeGroups, updateSmoothingLengthIterative,
  FindNeighbors, mpi::synchronizeHalos, Upsweep, star position/accretion, Timestep), *Gravity*,
  *Particle interaction* (everything else). Shares are averaged over ranks; bar height = runtime, the max over
  ranks of the per-rank total (equal to rank 0's: the ranks synchronise every step).
- `--sync-group` moves `Timestep` and `duTimestep` into a 4th group, *Waiting for the slowest rank* (grey, on top),
  and writes `OUT_STEM_sync*`. Rank 0 waits in `duTimestep` (the observables' `MPI_Reduce` runs just before it),
  the other ranks in `Timestep`; without this the wait lands in different groups on different ranks.
- `OUT_STEM.txt` also lists seconds per group as rank 0 / mean / max over ranks.
- `--absolute` (instead of `--runtime-heights`) writes `OUT_STEM[_sync]_absolute[_narrow].png`: one panel per phase
  group, 3D SFC / Hybrid-Dimensional SFC bar pair per rank count, time in s (mean over ranks) on a log y-axis shared by all panels,
  factor first/second variant above each pair. Use this to compare one group between builds; the stacked plots
  cannot (segments are shares, and the split/log axes distort them). Combines with `--sync-group` and `--narrow`
  (narrow: 2 panels per row, 6 in wide, 1.9 + 3.05 in per row high).
- Writes `OUT_STEM.txt` (shares, mean and rank 0) and `OUT_STEM_runtime.png`.
- With two variants whose ratio is > 3 (disk) each tall bar is cut once inside its largest segment (`//`, the time
  taken out written above the cut); all bars stay on one linear scale. `--split-axis` gives the old split y-axis.
- `--log` writes `OUT_STEM_runtime_log.png`: log axis, the bar top is the true runtime and segments split the
  drawn bar by share (inner edges are not readable on the axis). Preferred for disk.
- `--narrow` writes `OUT_STEM_runtime[_log]_narrow.png`: 6 × 7.2 in instead of 12 × 6.2, larger fonts, header wrapped,
  y label cut to its first line (the rest goes into the note above the graph), rank labels without "(GPUs)".
  Layout knobs are the `NARROW` dict at the top of the script; the default output is unchanged.
- Variant tick labels wrap at `tick_wrap` characters, never inside a word (17: "Hybrid- / Dimensional SFC", narrow 11:
  "Hybrid- / Dimensional / SFC"); in narrow plots the 'ranks' row moves down by the extra lines. Narrow cut bars put the
  `//` cut at `cut_at` = 40% of the segment instead of 50%, so the rotated removed time fits under the next label.
  Narrow bar pairs are spaced 0.46 apart with 0.07 between groups (was 0.42 / 0.15, same group pitch; changed
  2026-10-09) so "3D SFC" and "Hybrid-", which share the first tick line, stay ~12 px apart in the 4-group disk plots.
  The overlap checker still reports those as box overlaps; the ink does not touch.
- Narrow heights were cut ~15% on 2026-10-05 (only the plot area shrank: header and fonts are fixed in inches; the bottom margin is a figure fraction, 1-2 px less).
  To regenerate all 15 `*_narrow.png`: the commands in section 4/6 with `--narrow` (scaling: `--halos-only`;
  phase groups: `--runtime-heights [--log]` and `--absolute`, each with and without `--sync-group`; halo: `--spread`, `-o X_halo.png`).

## 4. Commands per case

### Wind-shock, Hybrid-Dimensional SFC vs 3D SFC (4/8/16 ranks, 199 counted steps)

Titled **Blob Test** in the figures (`--case "Blob Test"`, titles "Blob Test experiment, ..."; until 2026-10-08
"Wind-shock"). File and folder names keep `windshock`. All 14 `windshock_n400_mixd_vs_nonmixd_*.png` were
regenerated with it on 2026-10-08 (scaling `busiest` / `halos` / `halos_narrow`, the 8 phase-group plots and the 3
halo plots), so the scaling plots now also carry the new build names and halo colours.

Folders `windshock_r{R}_c72_s200_n{N}_gpu_BT0_SB0[_nonmixd]_prof_gpumpi`; N=200 -> 32,301,610 particles,
N=400 -> 258,412,816. Below for N=400 (for N=200 drop the `n400_` tag and use `n200` / 32301610).

```bash
N=400; NP=258412816; W=windshock_r%s_c72_s200_n${N}_gpu_BT0_SB0%s_prof_gpumpi
for r in 4 8 16; do n=$(printf $W $r _nonmixd); m=$(printf $W $r '')
  $P compare_runs_log.py nonmixd-BT0=$n/run.log:0:$n mixd-BT0=$m/run.log:0:$m \
    > windshock_mixd_vs_nonmixd_r${r}_s200_n${N}_results.txt; done
R=(windshock_mixd_vs_nonmixd_r{4,8,16}_s200_n${N}_results.txt)
$P plot_bt_scaling.py --busiest    --case "Blob Test" --particles $NP --variants nonmixd-BT0,mixd-BT0 \
  -o windshock_n400_mixd_vs_nonmixd_scaling_busiest.png "${R[@]}"
$P plot_bt_scaling.py --halos-only --case "Blob Test" --particles $NP --variants nonmixd-BT0,mixd-BT0 \
  -o windshock_n400_mixd_vs_nonmixd_scaling_halos.png "${R[@]}"
b=(); for r in 4 8 16; do b+=("nonmixd-BT0@$r=$(printf $W $r _nonmixd)" "mixd-BT0@$r=$(printf $W $r '')"); done
$P plot_phase_groups.py --runtime-heights --variants nonmixd-BT0,mixd-BT0 \
  --title "Blob Test experiment, 258.4M particles: time per phase group" \
  -o windshock_n400_mixd_vs_nonmixd_phase_groups "${b[@]}"   # add --narrow for the half-page version
```

### Disk (chain2x4), Hybrid-Dimensional SFC vs 3D SFC (8/16/32/64 ranks, 249 counted steps, 621,858,696 particles)

```bash
D=noah_disk_000_exa_zslab_w8_chain2x4.h5_DBM1_PM0_BT0_RLH5_DMLP1_%stasks_prof_NL%s_gpumpi
for r in 8 16 32 64; do n=$(printf $D $r _nonmixd); m=$(printf $D $r '')
  $P compare_runs_log.py nonmixd-BT0=$n/run.log:0:$n mixd-BT0=$m/run.log:0:$m \
    > disk_chain2x4_mixd_vs_nonmixd_r${r}_s500_results.txt; done
R=(disk_chain2x4_mixd_vs_nonmixd_r{8,16,32,64}_s500_results.txt)
A=(--case "Protoplanetary Disk" --particles 621858696 --variants nonmixd-BT0,mixd-BT0)
$P plot_bt_scaling.py --busiest    "${A[@]}" -o disk_chain2x4_mixd_vs_nonmixd_scaling_busiest.png "${R[@]}"
$P plot_bt_scaling.py --halos-only "${A[@]}" -o disk_chain2x4_mixd_vs_nonmixd_scaling_halos.png "${R[@]}"
$P plot_bt_scaling.py --busiest    --log-halos "${A[@]}" -o disk_chain2x4_mixd_vs_nonmixd_scaling_busiest_log.png "${R[@]}"
$P plot_bt_scaling.py --halos-only --log-halos "${A[@]}" -o disk_chain2x4_mixd_vs_nonmixd_scaling_halos_log.png "${R[@]}"
b=(); for r in 8 16 32 64; do b+=("nonmixd-BT0@$r=$(printf $D $r _nonmixd)" "mixd-BT0@$r=$(printf $D $r '')"); done
for L in "" --log; do for N in "" --narrow; do
  $P plot_phase_groups.py --runtime-heights $L $N --variants nonmixd-BT0,mixd-BT0 \
    --title "Protoplanetary Disk experiment, 621.9M particles: time per phase group" \
    -o disk_chain2x4_mixd_vs_nonmixd_phase_groups "${b[@]}"; done; done
```

### TDE, BT3 vs Hybrid-Dimensional SFC (BT0) (32 nodes / 128 ranks, 100 counted steps, 10,305,994,744 particles)

`tde_sim_debug.sh` (loop `for SPHEXA_BOUNDARY_TYPE in 0 3`) writes both runs into one slurm log
(BT0 = run 0, BT3 = run 1) and profiles to `tde-s166610-BT{0,3}_N<ntasks>_prof_NL_gpumpi`.

```bash
L=slurm-4965369.out; d0=tde-s166610-BT0_N128_prof_NL_gpumpi; d3=tde-s166610-BT3_N128_prof_NL_gpumpi
$P compare_runs_log.py BT3=$L:1:$d3 mixd-BT0=$L:0:$d0 > tde_bt3_vs_mixd_r128_s100_results.txt
$P plot_bt_scaling.py --halos-only --case TDE --particles 10305994744 --variants BT3,mixd-BT0 \
  -o tde_N128_bt3_vs_mixd_scaling_halos.png tde_bt3_vs_mixd_r128_s100_results.txt
$P plot_phase_groups.py --runtime-heights --variants BT3,mixd-BT0 \
  --title "TDE experiment, 10.3B particles: time per phase group" -o tde_N128_bt3_vs_mixd_phase_groups \
  BT3@128=$d3 mixd-BT0@128=$d0
```

Several rank counts: one results file per rank count (`..._r<ranks>_...`) and pass them all to
`plot_bt_scaling.py`; add `BT3@R=... mixd-BT0@R=...` pairs to `plot_phase_groups.py`.

### Quick BT0 vs BT3 runtime from a slurm log (no profile needed)

Sums `Total time for iteration` per run, skipping each run's first step:

```bash
awk '/Running with SPHEXA_BOUNDARY_TYPE/{bt=$NF; sub(/.*=/,"",bt); k=0}
     /=== Total time for iteration/{match($0,/\) ([0-9.]+)s/,m); if(k++) {s[bt]+=m[1]; n[bt]++}}
     END{for(b in s) printf "BT%s %d steps %.1fs\n",b,n[b],s[b]; if(s[0]) printf "BT3/BT0 = %.3f\n",s[3]/s[0]}' slurm-XXXX.out
```

## 5. Particle-to-rank distribution (wind-shock N=400)

```bash
for v in mixd nonmixd; do for r in 4 8 16; do
  s=$([ $v = nonmixd ] && echo _nonmixd); l=$([ $v = mixd ] && echo "Hybrid-Dimensional SFC" || echo "3D SFC")
  $P plot_rank_distribution_sampled.py windshock_r${r}_c72_s200_n400_gpu_BT0_SB0${s}_prof_gpumpi \
    --tag windshock_n400_$v --label "$l" &
done; done; wait
```

- Writes `<tag>_5k_<R>ranks.png/.html` (`--subsample` particles per rank, default 5,000, sampled at random
  byte offsets of `coords_rank*.txt`), prints per-rank particle/halo counts from `profile.h5`.
- Dots: PNG `s=2`, `alpha=0.8`; HTML `size=2.5`, `opacity=0.8` (passed by `plot_rank_distribution_sampled.py`;
  standalone `plot_rank_distribution.py` keeps `s=0.3`/`size=1.2`, alpha 0.5). Changed on 2026-10-05 from 20k samples
  with the small translucent dots; the old `windshock_n400_*_20k_*` files were kept.
- Depth order (2026-10-05): all ranks are one scatter coloured per particle, so matplotlib depth-sorts every point and
  the particles nearest the camera (default view elev 30, azim -60: camera on the +x, -y, +z side) are drawn last.
  Before, one scatter per rank painted higher ranks over lower ones regardless of depth. Regenerated with it so far:
  only the 16-rank plots (`windshock_n400_*` and `disk_chain2x4_*`, 5k/7.5k/10k/50k/100k).
- Title: "Particle distribution to ranks, LABEL (N total, M shown)", centred over the plot; a title wider than the
  plot is shifted left to end at the plot's right edge, or centred on plot + legend if wider than both (the long
  "Hybrid-Dimensional SFC" titles), and the crop is measured at the saved 200 dpi.
- Cross section (2026-10-05, opt-in `--section-y W`, PNG only): below the 3D plot, an x-z panel of the sampled
  particles within W of the middle y of the run's computational domain (first `Computational domain:`
  line of `DIR/run.log`; the sampled box, with a warning, if there is none), seen from -y (lowest-y dots drawn last),
  at true scale and as wide as the 3D plot, titled "Cross section at y = Y (±W, K particles)"; the figure grows
  downwards, the 3D plot is unchanged. Only made for the disk 16-rank 100k plots, with `W = 1` (~80k particles, 5% of the
  sample; both sides of the ring around the central hole). Both runs have the same domain, so both cut at
  y = -3.77 (the sampled box's middle, used at first, depended on which outliers were sampled: -2.8 vs -4.1):
  ```bash
  for v in mixd nonmixd; do
    s=$([ $v = nonmixd ] && echo _nonmixd); l=$([ $v = mixd ] && echo "Hybrid-Dimensional SFC" || echo "3D SFC")
    $P plot_rank_distribution_sampled.py noah_disk_000_exa_zslab_w8_chain2x4.h5_DBM1_PM0_BT0_RLH5_DMLP1_16tasks_prof_NL${s}_gpumpi \
      --tag disk_chain2x4_$v --label "$l" --subsample 100000 --section-y 1 &
  done; wait
  ```
- Legend: one column left of the plot, same height; print-friendly 16-colour palette for <= 16 ranks
  (`PRINT_PALETTE` in `plot_rank_distribution.py`), HSV fallback above. The PNG is cropped to plot + legend + title.

## 6. Halo imbalance across ranks (wind-shock N=400, disk chain2x4)

Per run: H_r = `numHalos` of rank r summed over the counted steps (first logged step skipped, as in the results
tables); bar = max_r H_r / mean_r H_r (1 = perfect balance, dashed line). The small label under each factor is
the busiest rank's mean halos per step (matches the log's `Halos` line).

```bash
W=windshock_r%s_c72_s200_n400_gpu_BT0_SB0%s_prof_gpumpi
b=(); for r in 4 8 16; do b+=("nonmixd-BT0@$r=$(printf $W $r _nonmixd)" "mixd-BT0@$r=$(printf $W $r '')"); done
$P plot_halo_imbalance.py --variants nonmixd-BT0,mixd-BT0 \
  --title "Blob Test experiment, 258.4M particles: halo imbalance across ranks (max / mean)" \
  -o windshock_n400_mixd_vs_nonmixd_halo_imbalance.png "${b[@]}"
D=noah_disk_000_exa_zslab_w8_chain2x4.h5_DBM1_PM0_BT0_RLH5_DMLP1_%stasks_prof_NL%s_gpumpi
b=(); for r in 8 16 32 64; do b+=("nonmixd-BT0@$r=$(printf $D $r _nonmixd)" "mixd-BT0@$r=$(printf $D $r '')"); done
$P plot_halo_imbalance.py --variants nonmixd-BT0,mixd-BT0 \
  --title "Protoplanetary Disk experiment, 621.9M particles: halo imbalance across ranks (max / mean)" \
  -o disk_chain2x4_mixd_vs_nonmixd_halo_imbalance.png "${b[@]}"
```

`--spread` (same run specs) draws the spread instead: bar = mean over ranks of H_r, whisker = min..max over ranks
(linear axis), label = spread max_r H_r - min_r H_r (M or B halos); one bracket per rank count gives the mean ratio
3D SFC / Hybrid-Dimensional SFC and the spread difference 3D SFC - Hybrid-Dimensional SFC (+ = Hybrid-Dimensional SFC ranks closer together). `-o X_halo.png` writes `X_halo_spread.png`;
add `--narrow` for `X_halo_spread_narrow.png` (6 × 6.3 in, larger fonts). Titles used: the full-width spread plots
"Blob Test experiment, 258.4M particles: halo spread across ranks" and "Protoplanetary Disk experiment, 621.9M
particles: halo spread across ranks"; the narrow ones the "... halo imbalance across ranks (max / mean)" titles above. The run also prints a table of
mean/min/max/max-min/max-mean per run. Disk: 3D SFC's spread is 0.6-4.2B halos larger than Hybrid-Dimensional SFC's (mean
5.3-6.9x); wind-shock: -89.5M / -89.8M / +35.5M at 4 / 8 / 16 ranks (mean 1.3-1.9x).

## Gotchas

- Two jobs writing the same run folder interleave `run.log`: check `grep -c 'process(es)' DIR/run.log` is 1.
- A running job can delete `profile.h5` of an older run in the same folder; check it exists before plotting.
- `sacct -B` (batch script) is not available through husk; `squeue`/`scontrol` may time out - read the
  slurm `.out` files instead. Slurm keeps the script as submitted, so later edits to `tde_sim_debug.sh`
  do not apply to queued jobs.
- 3D SFC (nonMixD) TDE runs out of GPU memory at 16 nodes / 64 ranks (thrust `bad_alloc`); use >= 32 nodes.
- `duTimestep` on rank 0 includes the observables `MPI_Reduce`, so it absorbs waits for the slowest rank;
  large bursty `duTimestep` usually means load imbalance or a slow node, not the timestep itself.
- Old 3-variant plots (`*_scaling_busiest.png`, `*_phase_groups_runtime.png` without `mixd_vs_nonmixd`) are
  from the default `--variants` and were left in place.
