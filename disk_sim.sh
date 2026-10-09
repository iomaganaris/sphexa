#!/usr/bin/env bash
#SBATCH --job-name=disk_sim_mixd
#SBATCH -N2
#SBATCH --ntasks-per-node=4
#SBATCH -c72
#SBATCH -A csstaff
#SBATCH --uenv=prgenv-gnu/26.3:v1@santis
#SBATCH --view=default
#SBATCH --time=01:30:00
#SBATCH --partition normal

set -o pipefail

# export CUDA_BUFFER_PAGE_IN_THRESHOLD_MS=0.001
# export FI_CXI_SAFE_DEVMEM_COPY_THRESHOLD=0
# export FI_CXI_RX_MATCH_MODE=software
# export FI_MR_CACHE_MONITOR=disabled
export MPICH_GPU_SUPPORT_ENABLED=1

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

# SPHEXA_DISK_BOUND_MULTIPLIER values (see cstone::Box::applyEnvDiskBoundMultiplier in
# domain/include/cstone/sfc/box.hpp): scales the loaded x/y domain about its center.
# Only swept when SPHEXA_BOUNDARY_TYPE == 2 (fixed boundary), since widening the domain
# only matters for a boundary condition that clamps/reflects particles at the box edge.
SPHEXA_DISK_BOUND_MULTIPLIER_VALUES=(1)

PARTICLE_MULTIPLIER=0

export SPHEXA_SPACIAL_BINS=0

# SPHEXA_WRITE_FREQ controls HDF5 checkpoint output via sphexa-cuda's -w flag:
#   0 = disabled (no checkpoint writes at all; avoids the I/O spike on write iterations)
#   N > 0 = write every N iterations
SPHEXA_WRITE_FREQ=0

# ENABLE_PROFILE_VALUES: each run is repeated for every value listed here. ENABLE_PROFILE
# controls the per-rank profile.h5 dump and the halo-count analysis on it:
#   0 = disabled (no --profile / -op flags)
#   1 = enabled (print_rank_halo_counts.py runs at the end on all profiled runs)
ENABLE_PROFILE_VALUES=(1)

# DISABLE_NEIGHBOR_LISTS controls sphexa-cuda's --disable-neighbor-lists flag:
#   0 = neighbor lists enabled (flag omitted)
#   1 = neighbor lists disabled (flag passed)
DISABLE_NEIGHBOR_LISTS=0

# FileSplitInit (main/src/init/file_init.hpp) requires numSplits >= 1 and is only
# selected when --init has a comma-separated suffix; omit the suffix entirely for
# PARTICLE_MULTIPLIER < 1 instead of passing a value FileSplitInit would reject.
# SNAPSHOT_FILE=/capstor/scratch/cscs/ioannmag/CORNERSTONE/sphexa/snapshot_000_exa.hdf5 # bz 19
# SNAPSHOT_FILE=/capstor/scratch/cscs/ioannmag/CORNERSTONE/sphexa/snapshot_000_exa_zslab10.hdf5
# SNAPSHOT_FILE=/capstor/scratch/cscs/ioannmag/CORNERSTONE/sphexa/snapshot_000_exa_zslab_w17.38929.hdf5 # bz 18
# SNAPSHOT_FILE=/capstor/scratch/cscs/ioannmag/CORNERSTONE/sphexa/snapshot_000_exa_zslab_bz17_max.hdf5 # bz 17 - wrong, still 18
# SNAPSHOT_FILE=/capstor/scratch/cscs/ioannmag/CORNERSTONE/sphexa/snapshot_000_exa_zslab_w8.hdf5 # bz 17
# SNAPSHOT_FILE=/capstor/scratch/cscs/ioannmag/CORNERSTONE/sphexa/snapshot_000_exa_zslab_w8_cubicBT0.hdf5 # bz 17 same as above but it should create an isotropic box when splitting
# SNAPSHOT_FILE=/capstor/scratch/cscs/ioannmag/CORNERSTONE/sphexa/snapshot_000_exa_zslab_w8_cubicBT0_origbox.hdf5 # bz = 17 -> for use with PARTICLE_MULTIPLIER
# SNAPSHOT_FILE=/capstor/scratch/cscs/ioannmag/CORNERSTONE/sphexa/snapshot_000_exa_zslab_w8_PM8split.h5
# SNAPSHOT_FILE=/capstor/scratch/cscs/ioannmag/CORNERSTONE/sphexa/snapshot_000_exa_zslab_w8_chain2x3.h5
SNAPSHOT_FILE=/capstor/scratch/cscs/ioannmag/CORNERSTONE/sphexa/snapshot_000_exa_zslab_w8_chain2x4.h5
if (( PARTICLE_MULTIPLIER < 1 )); then
    INIT_ARG=$SNAPSHOT_FILE
else
    INIT_ARG=$SNAPSHOT_FILE,${PARTICLE_MULTIPLIER}
fi

export SPHEXA_WRITE_RANK_COORDS=1

export STEPS=500

# Extract the tag between "snapshot_" and ".hdf5" from SNAPSHOT_FILE's basename
# (e.g. "000_exa_zslab_w8_cubicBT0") so output dirs identify which snapshot was used.
SNAPSHOT_TAG=$(basename "$SNAPSHOT_FILE")
SNAPSHOT_TAG=${SNAPSHOT_TAG#snapshot_}
SNAPSHOT_TAG=${SNAPSHOT_TAG%.hdf5}

LOG_FILES=()
PROFILE_DIRS=()

export SPHEXA_MIXD_MAX_LEVEL_DIFF=21

echo "Running with init $INIT_ARG on $SLURM_NNODES nodes with updates to spliting"

for SPHEXA_BOUNDARY_TYPE in 0; do
    export SPHEXA_BOUNDARY_TYPE

    # if (( SPHEXA_BOUNDARY_TYPE == 2 )); then
    DISK_BOUND_MULTIPLIERS=("${SPHEXA_DISK_BOUND_MULTIPLIER_VALUES[@]}")
        # export PARTICLE_MULTIPLIER=0 # split init not fixed yet for that
    # else
    #     DISK_BOUND_MULTIPLIERS=(1)
    # fi

    for SPHEXA_DISK_BOUND_MULTIPLIER in "${DISK_BOUND_MULTIPLIERS[@]}"; do
        export SPHEXA_DISK_BOUND_MULTIPLIER

        for SPHEXA_REMOVAL_LIMIT_H in "${SPHEXA_REMOVAL_LIMIT_H_VALUES[@]}"; do
            export SPHEXA_REMOVAL_LIMIT_H

            for ENABLE_PROFILE in "${ENABLE_PROFILE_VALUES[@]}"; do
                echo "Running MixD with SPHEXA_BOUNDARY_TYPE=$SPHEXA_BOUNDARY_TYPE SPHEXA_DISK_BOUND_MULTIPLIER=$SPHEXA_DISK_BOUND_MULTIPLIER SPHEXA_REMOVAL_LIMIT_H=$SPHEXA_REMOVAL_LIMIT_H SPECIAL_BINS=$SPHEXA_SPACIAL_BINS SPHEXA_WRITE_RANK_COORDS=$SPHEXA_WRITE_RANK_COORDS ENABLE_PROFILE=$ENABLE_PROFILE"
                if [ ${SPHEXA_SPACIAL_BINS} -eq 1 ]; then
                    SPECIAL_BINS_FLAG="SB15_"
                else
                    SPECIAL_BINS_FLAG=""
                fi
                if (( ENABLE_PROFILE )); then
                    PROFILE_FLAG="_prof"
                else
                    PROFILE_FLAG="_noprof"
                fi
                NEIGHBOR_LISTS_ARGS=()
                if (( DISABLE_NEIGHBOR_LISTS )); then
                    NEIGHBOR_LISTS_ARGS=(--disable-neighbor-lists)
                    NEIGHBOR_LISTS_FLAG="_noNL"
                else
                    NEIGHBOR_LISTS_FLAG="_NL"
                fi
                OUTPUT_DIR=$(pwd)/noah_disk_${SNAPSHOT_TAG}_DBM${SPHEXA_DISK_BOUND_MULTIPLIER}_PM${PARTICLE_MULTIPLIER}_BT${SPHEXA_BOUNDARY_TYPE}_RLH${SPHEXA_REMOVAL_LIMIT_H}_DMLP${SPHEXA_DISK_BOUND_MULTIPLIER}_${SPECIAL_BINS_FLAG}${SLURM_NTASKS}tasks${PROFILE_FLAG}${NEIGHBOR_LISTS_FLAG}_gpumpi
                mkdir -p $OUTPUT_DIR
                OUTPUT_FILE=$OUTPUT_DIR/output
                rm -f $OUTPUT_FILE
                RUN_LOG=$OUTPUT_DIR/run.log
                LOG_FILES+=("$RUN_LOG")
                PROFILE_ARGS=()
                if (( ENABLE_PROFILE )); then
                    PROFILE_DIRS+=("$OUTPUT_DIR")
                    PROFILE_ARGS=(--profile -op "$OUTPUT_DIR/profile")
                fi
                # --profile enables propagator->writeMetrics() which writes per-rank numHalos,
                # numParticles, assignment, etc. to profile.h5 at the end of the run; -op puts it in
                # $OUTPUT_DIR/profile.h5 so concurrent jobs launched from this directory don't clobber it.
                # See main/src/propagator/ipropagator.hpp:187-190 (logDomainStats) and
                # main/src/util/timer.hpp:79-117 (writeTimings/perfStats).
                OMP_NUM_THREADS=64 srun --mpi=cray_shasta ./mps_wrapper.sh ./build_gpu_release/main/src/sphexa/sphexa-cuda "${NEIGHBOR_LISTS_ARGS[@]}" --init $INIT_ARG --prop std-disk -s ${STEPS} -w $SPHEXA_WRITE_FREQ "${PROFILE_ARGS[@]}" -o $OUTPUT_FILE 2>&1 | tee "$RUN_LOG"
            done
        done
    done
done

echo
echo "===== Last 9 lines of each run ====="
for RUN_LOG in "${LOG_FILES[@]}"; do
    echo
    echo "--- $RUN_LOG ---"
    tail -n 9 "$RUN_LOG"
done

if (( ${#PROFILE_DIRS[@]} )); then
    echo
    echo "===== Per-rank halo counts from profile.h5 ====="
    $(pwd)/venv_sphexa_new/bin/python3 ./print_rank_halo_counts.py "${PROFILE_DIRS[@]}"
fi
