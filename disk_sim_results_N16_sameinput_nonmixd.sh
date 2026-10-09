#!/usr/bin/env bash
#SBATCH --job-name=disk_sim_nonmixd
#SBATCH -N16
#SBATCH --ntasks-per-node=4
#SBATCH -c72
#SBATCH -A csstaff
#SBATCH --uenv=prgenv-gnu/26.3:v1@santis
#SBATCH --view=default
#SBATCH --time=01:00:00
#SBATCH --partition normal
#SBATCH --exclude=nid005669

set -o pipefail

# SPHEXA_BOUNDARY_TYPE values (see cstone::BoundaryType in domain/include/cstone/sfc/box.hpp):
#   0 = open
#   1 = periodic
#   2 = fixed
#   3 = cubic_open

# SPHEXA_REMOVAL_LIMIT_H values (see disk::StarData::removal_limit_h in
# physics/Disk/include/star_data.hpp): overrides the ejecta-removal smoothing-length
# threshold from snapshot_000_exa.hdf5's star::removal_limit_h attribute without editing
# the file. Swept between the snapshot's current stored value (5, effectively disabled --
# see PLAN_removal_limit_h_analysis.md) and the range recommended by
# scripts/analysis/analyze_removal_limit_h.py (0.5915 to 0.907).
SPHEXA_REMOVAL_LIMIT_H_VALUES=(5)

# the input snapshot already contains the particles split by 16 (created by make_split_snapshot_N16.sh),
# so no particle splitting at initialization
PARTICLE_MULTIPLIER=0

# SPHEXA_DISK_BOUND_MULTIPLIER is only relevant for SPHEXA_BOUNDARY_TYPE == 2 (fixed
# boundary); this sweep only runs BT 0 and 3, so it stays fixed at 1 (no-op scaling).
# SPHEXA_DISK_BOUND_MULTIPLIER=1

# SPHEXA_SPACIAL_BINS=1 assigns ranks by X/Y-plane columns of the SFC instead of plain uniform
# SFC bins (see cstone::spacialBins in domain/include/cstone/domain/domaindecomp.hpp)
export SPHEXA_SPACIAL_BINS=0

# SPHEXA_WRITE_FREQ controls HDF5 checkpoint output via sphexa-cuda's -w flag:
#   0 = disabled (no checkpoint writes at all; avoids the I/O spike on write iterations)
#   N > 0 = write every N iterations
SPHEXA_WRITE_FREQ=0

SNAPSHOT_FILE=/capstor/scratch/cscs/ioannmag/CORNERSTONE/sphexa/snapshot_000_exa_zslab_w8_PM16split.h5
if [ ! -f "$SNAPSHOT_FILE" ]; then
    echo "Snapshot $SNAPSHOT_FILE not found, create it with make_split_snapshot_N16.sh" >&2
    exit 1
fi
if (( PARTICLE_MULTIPLIER < 1 )); then
    INIT_ARG=$SNAPSHOT_FILE
else
    INIT_ARG=$SNAPSHOT_FILE,${PARTICLE_MULTIPLIER}
fi

# Extract the tag between "snapshot_" and ".hdf5" from SNAPSHOT_FILE's basename
# (e.g. "000_exa_zslab_w8_cubicBT0") so output dirs identify which snapshot was used.
SNAPSHOT_TAG=$(basename "$SNAPSHOT_FILE")
SNAPSHOT_TAG=${SNAPSHOT_TAG#snapshot_}
SNAPSHOT_TAG=${SNAPSHOT_TAG%.hdf5}
SNAPSHOT_TAG=${SNAPSHOT_TAG%.h5}

# -s is the last iteration, not a count: the snapshot restarts after iteration 1, so STEPS = 1 + iterations to run
NUM_ITERATIONS=500

# SPHEXA_PROFILE_HALOS=1 passes --profile to sphexa-cuda and prints per-rank halo counts from profile.h5 at the end
SPHEXA_PROFILE_HALOS=${SPHEXA_PROFILE_HALOS:-0}
if [ "$SPHEXA_PROFILE_HALOS" = "1" ]; then
    PROFILE_FLAG=--profile
    NUM_ITERATIONS=1
else
    PROFILE_FLAG=
fi

STEPS=$((NUM_ITERATIONS + 1))

LOG_FILES=()
PROFILE_DIRS=()

echo "Running with init $INIT_ARG on $SLURM_NNODES nodes with updates to spliting"

export SPHEXA_DISK_BOUND_MULTIPLIER

export SPHEXA_WRITE_RANK_COORDS=1

for SPHEXA_BOUNDARY_TYPE in 0; do
    export SPHEXA_BOUNDARY_TYPE

    for SPHEXA_REMOVAL_LIMIT_H in "${SPHEXA_REMOVAL_LIMIT_H_VALUES[@]}"; do
        export SPHEXA_REMOVAL_LIMIT_H

        echo "Running with SPHEXA_BOUNDARY_TYPE=$SPHEXA_BOUNDARY_TYPE SPHEXA_DISK_BOUND_MULTIPLIER=$SPHEXA_DISK_BOUND_MULTIPLIER SPHEXA_REMOVAL_LIMIT_H=$SPHEXA_REMOVAL_LIMIT_H SPHEXA_SPACIAL_BINS=$SPHEXA_SPACIAL_BINS"
        OUTPUT_DIR=$(pwd)/noah_disk_${SNAPSHOT_TAG}_BT${SPHEXA_BOUNDARY_TYPE}_RLH${SPHEXA_REMOVAL_LIMIT_H}_DMLP${SPHEXA_DISK_BOUND_MULTIPLIER}_SB${SPHEXA_SPACIAL_BINS}_${SLURM_NTASKS}tasks_nonmixd
        mkdir -p $OUTPUT_DIR
        OUTPUT_FILE=$OUTPUT_DIR/output
        rm -f $OUTPUT_FILE
        RUN_LOG=$OUTPUT_DIR/run.log
        LOG_FILES+=("$RUN_LOG")
        PROFILE_DIRS+=("$OUTPUT_DIR")
        OMP_NUM_THREADS=64 srun --mpi=cray_shasta ./mps_wrapper.sh ./build_gpu_release_nonmixd/main/src/sphexa/sphexa-cuda --init $INIT_ARG --prop std-disk -s ${STEPS} -w $SPHEXA_WRITE_FREQ $PROFILE_FLAG -o $OUTPUT_FILE 2>&1 | tee "$RUN_LOG"
        if [ "$SPHEXA_PROFILE_HALOS" = "1" ]; then
            mv profile.h5 $OUTPUT_DIR/profile.h5
        fi
    done
done

echo
echo "===== Last 9 lines of each run ====="
for RUN_LOG in "${LOG_FILES[@]}"; do
    echo
    echo "--- $RUN_LOG ---"
    tail -n 9 "$RUN_LOG"
done

if [ "$SPHEXA_PROFILE_HALOS" = "1" ]; then
    echo
    echo "===== Per-rank halo counts from profile.h5 ====="
    $(pwd)/venv_sphexa_new/bin/python3 ./print_rank_halo_counts.py "${PROFILE_DIRS[@]}"
fi
