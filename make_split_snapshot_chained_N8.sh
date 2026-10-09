#!/usr/bin/env bash
#SBATCH --job-name=chain_split
#SBATCH -N8
#SBATCH --ntasks-per-node=4
#SBATCH -c72
#SBATCH -A csstaff
#SBATCH --uenv=prgenv-gnu/26.3:v1@santis
#SBATCH --view=default
#SBATCH --time=00:30:00
#SBATCH --partition normal

# Builds a 16x particle-split snapshot in four stages instead of one, relaxing the particles
# for RELAX_STEPS time steps between splits:
#
#   snapshot_000_exa_zslab_w8.hdf5 --x2--> chain2x1 --x2--> chain2x2 --x2--> chain2x3 --x2--> chain2x4
#        38,895,141 particles            77.8 M          155.6 M         311.2 M         622.3 M
#
# Compare make_split_snapshot_N16.sh, which applies the same total 16x in a single split
# (snapshot_000_exa_zslab_w8_PM16split.h5).
#
# By default the whole chain is rebuilt from the original (START_STAGE=1, FORCE=1). Set
# START_STAGE=k to resume from an existing chain2x(k-1) and leave the earlier stages untouched.
#
# The dump is written *before* integrate() (main/src/sphexa/sphexa.cpp:288-296) and FileSplitInit
# starts at d.iteration = 1, so a dump at iteration k holds the state after k-1 integrations:
# RELAX_STEPS real steps therefore need -s/-w of RELAX_STEPS+1.
#
# FileSplitInit divides minDt by 100*numSplits at every split, so each stage starts with tiny
# steps; RELAX_STEPS=250 is what gives the particles time to actually move (stage 1 reaches
# t ~12.6).
#
# That movement puffs the disk up: after 250 steps stage 1 spans z in [-5.94, +6.74] against the
# original [-3.63, +4.37]. After each stage filter_z_domain.py therefore drops the particles
# outside the original z range (~0.04% at stage 1) and rewrites the output in place; without it the
# stamp below would have to widen z enough to change the bit depths, and would refuse.
#
# After the filter the box is stamped back to the original snapshot's by stamp_original_box.py.
# This is not cosmetic: sphexa writes domain.box(), which under BT3 is the cubic box
# (z extended to ~89), and FileSplitInit derives its mixD Hilbert keys from the box it reads.
# Without the stamp, stage 1 would split in the slab key space {21,21,17} and the later stages in
# the isotropic {21,21,21} -- one chain, two different geometries. The stamp also verifies the
# bit depths are unchanged and aborts the chain if they are not.
#
# Expect the final particle count to be under 16 x 38,895,141 = 622,322,256: std-disk removes
# ejecta at each step, and the z filter removes the particles that left the slab.

set -euo pipefail

ORIGINAL=/capstor/scratch/cscs/ioannmag/CORNERSTONE/sphexa/snapshot_000_exa_zslab_w8.hdf5
SPLIT=2                             # PARTICLE_MULTIPLIER per stage; SPLIT^STAGES is the total
STAGES=${STAGES:-4}                 # index of the last stage to produce
START_STAGE=${START_STAGE:-4}       # first stage to run; >1 resumes from that stage's predecessor
RELAX_STEPS=250                     # integration steps between splits
DUMP_AT=$((RELAX_STEPS + 1))
FORCE=${FORCE:-1}                   # 1 to overwrite stage outputs that already exist

# The chain was built by this commit. Rebuilding sphexa between stages would mean chain2x4 comes
# from different code than chain2x1..3, so refuse unless explicitly waived.
EXPECT_COMMIT=1ab7004d
SKIP_COMMIT_CHECK=${SKIP_COMMIT_CHECK:-0}

PYTHON=$(pwd)/venv_sphexa_new/bin/python3
SPHEXA=$(pwd)/build_gpu_release/main/src/sphexa/sphexa-cuda

export SPHEXA_BOUNDARY_TYPE=3
export SPHEXA_REMOVAL_LIMIT_H=5
export SPHEXA_SPACIAL_BINS=0
export SPHEXA_DISK_BOUND_MULTIPLIER=1
export SPHEXA_WRITE_RANK_COORDS=0
export SPHEXA_MIXD_MAX_LEVEL_DIFF=21
# left at its default 0.5849625 to match disk_sim.sh and every benchmark run; unset it here so an
# inherited value can't silently change the split geometry (and the stamp script's bit check)
unset SPHEXA_MIXD_BIAS

BASE=$(basename "$ORIGINAL")
BASE=${BASE%.hdf5}

stage_output() { printf '%s/%s_chain%sx%s.h5' "$(pwd)" "$BASE" "$SPLIT" "$1"; }

# ---------------------------------------------------------------- pre-flight

if ((START_STAGE < 1 || START_STAGE > STAGES)); then
    echo "START_STAGE=$START_STAGE is not in 1..$STAGES" >&2
    exit 1
fi

if ((START_STAGE == 1)); then
    IN=$ORIGINAL
else
    IN=$(stage_output $((START_STAGE - 1)))
fi

if [[ ! -s $IN ]]; then
    echo "input for stage $START_STAGE is missing or empty: $IN" >&2
    exit 1
fi

# fail before burning an srun if a stage would clobber an existing snapshot
CLOBBER=()
for ((STAGE = START_STAGE; STAGE <= STAGES; STAGE++)); do
    OUT=$(stage_output "$STAGE")
    [[ -e $OUT ]] && CLOBBER+=("$OUT")
done
if ((${#CLOBBER[@]} > 0)) && ((FORCE != 1)); then
    echo "these stage outputs already exist; re-run with FORCE=1 to overwrite:" >&2
    printf '  %s\n' "${CLOBBER[@]}" >&2
    exit 1
fi

if ((SKIP_COMMIT_CHECK != 1)); then
    if ! command -v strings >/dev/null; then
        echo "strings(1) not available; re-run with SKIP_COMMIT_CHECK=1 to bypass" >&2
        exit 1
    fi
    # grep -c rather than -q: with pipefail, -q exits early and strings dies of SIGPIPE (141),
    # which would make the pipeline "fail" even on a match
    FOUND=$(strings -a "$SPHEXA" | grep -cxF "$EXPECT_COMMIT") || FOUND=0
    if ((FOUND == 0)); then
        echo "$SPHEXA does not carry commit $EXPECT_COMMIT, which built chain2x1..3." >&2
        echo "Continuing would splice a differently-built stage into the chain." >&2
        echo "Rebuild at $EXPECT_COMMIT, or re-run with SKIP_COMMIT_CHECK=1 if that is intended." >&2
        exit 1
    fi
fi

# ---------------------------------------------------------------- run

echo "Chaining splits of x$SPLIT up to stage $STAGES (total x$((SPLIT ** STAGES))) from $ORIGINAL"
echo "  running stages $START_STAGE..$STAGES, starting from $(basename "$IN")"
echo "  $RELAX_STEPS relaxation steps per stage, dumping at iteration $DUMP_AT"
echo "  $SLURM_NNODES nodes, $SLURM_NTASKS ranks, BT$SPHEXA_BOUNDARY_TYPE, sphexa $EXPECT_COMMIT"

WRITTEN=()

for ((STAGE = START_STAGE; STAGE <= STAGES; STAGE++)); do
    OUT=$(stage_output "$STAGE")
    WORK=$(pwd)/chain_stage${STAGE}_BT${SPHEXA_BOUNDARY_TYPE}_${SLURM_NTASKS}tasks
    mkdir -p "$WORK"
    # the HDF5 writer appends a step to an existing file and FileSplitInit reads the last step
    rm -f "$WORK/snapshot.h5"

    echo
    echo "===== stage $STAGE/$STAGES: $(basename "$IN") x$SPLIT -> $(basename "$OUT") ====="

    # -op is required even though --profile is absent: profiling is enabled implicitly whenever
    # file output is (sphexa.cpp:169), so without it all stages would collide on ./profile.h5.
    # -o points into $WORK so that constants.txt lands there too, then the dump is renamed.
    OMP_NUM_THREADS=64 srun --mpi=cray_shasta ./mps_wrapper.sh "$SPHEXA" \
        --disable-neighbor-lists --init "${IN},${SPLIT}" --prop std-disk \
        -s "$DUMP_AT" -w "$DUMP_AT" -op "$WORK/profile" -o "$WORK/snapshot.h5" \
        2>&1 | tee "$WORK/run.log"

    mv -f "$WORK/snapshot.h5" "$OUT"
    # tee'd separately: the pipeline above has already closed, so without this the filter's and
    # stamp's reports would go to the job file instead of run.log.
    # OPENBLAS/OMP pinned to 1: both are chunked h5py I/O with no BLAS work, and the default
    # thread pool can hit RLIMIT_NPROC on a busy node.
    # The filter runs first so that the stamp sees only particles inside the original z range.
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
        "$PYTHON" ./filter_z_domain.py "$ORIGINAL" "$OUT" 2>&1 | tee -a "$WORK/run.log"
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
        "$PYTHON" ./stamp_original_box.py "$ORIGINAL" "$OUT" 2>&1 | tee -a "$WORK/run.log"

    WRITTEN+=("$OUT")
    IN=$OUT
done

echo
echo "===== chain complete ====="
for ((STAGE = 1; STAGE <= STAGES; STAGE++)); do
    OUT=$(stage_output "$STAGE")
    MARK=""
    for W in "${WRITTEN[@]}"; do [[ $W == "$OUT" ]] && MARK=" (new)"; done
    if [[ -e $OUT ]]; then
        printf '  %10s  %s%s\n' "$(du -h "$OUT" | cut -f1)" "$OUT" "$MARK"
    else
        printf '  %10s  %s\n' "missing" "$OUT"
    fi
done
