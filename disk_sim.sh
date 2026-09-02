#!/usr/bin/env bash
#SBATCH --job-name=tde_sim
#SBATCH -N1
#SBATCH --ntasks-per-node=4
#SBATCH -c72
#SBATCH -A csstaff
#SBATCH --uenv=prgenv-gnu/26.3:v1@santis
#SBATCH --view=default
#SBATCH --time=04:00:00
#SBATCH --partition normal

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

PARTICLE_MULTIPLIER=0

# FileSplitInit (main/src/init/file_init.hpp) requires numSplits >= 1 and is only
# selected when --init has a comma-separated suffix; omit the suffix entirely for
# PARTICLE_MULTIPLIER < 1 instead of passing a value FileSplitInit would reject.
SNAPSHOT_FILE=/capstor/scratch/cscs/ioannmag/CORNERSTONE/sphexa/snapshot_000_exa.hdf5
if (( PARTICLE_MULTIPLIER < 1 )); then
    INIT_ARG=$SNAPSHOT_FILE
else
    INIT_ARG=$SNAPSHOT_FILE,${PARTICLE_MULTIPLIER}
fi

LOG_FILES=()

for SPHEXA_BOUNDARY_TYPE in 0 3; do
    export SPHEXA_BOUNDARY_TYPE

    for SPHEXA_REMOVAL_LIMIT_H in "${SPHEXA_REMOVAL_LIMIT_H_VALUES[@]}"; do
        export SPHEXA_REMOVAL_LIMIT_H

        echo "Running with SPHEXA_BOUNDARY_TYPE=$SPHEXA_BOUNDARY_TYPE SPHEXA_REMOVAL_LIMIT_H=$SPHEXA_REMOVAL_LIMIT_H"
        OUTPUT_DIR=$(pwd)/noah_disk_PM${PARTICLE_MULTIPLIER}_BT${SPHEXA_BOUNDARY_TYPE}_RLH${SPHEXA_REMOVAL_LIMIT_H}
        mkdir -p $OUTPUT_DIR
        OUTPUT_FILE=$OUTPUT_DIR/output
        rm -f $OUTPUT_FILE
        RUN_LOG=$OUTPUT_DIR/run.log
        LOG_FILES+=("$RUN_LOG")
        OMP_NUM_THREADS=64 srun --mpi=cray_shasta ./mps_wrapper.sh ./build_gpu_release/main/src/sphexa/sphexa-cuda --init $INIT_ARG --prop std-disk -s 100 -w 100 -o $OUTPUT_FILE 2>&1 | tee "$RUN_LOG"
    done
done

echo
echo "===== Last 9 lines of each run ====="
for RUN_LOG in "${LOG_FILES[@]}"; do
    echo
    echo "--- $RUN_LOG ---"
    tail -n 9 "$RUN_LOG"
done
