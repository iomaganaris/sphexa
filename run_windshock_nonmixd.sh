#!/usr/bin/env bash
#SBATCH --job-name=windshock_nonmixd
#SBATCH -N1
#SBATCH --ntasks-per-node=4
#SBATCH -c72
#SBATCH -A csstaff
#SBATCH --uenv=prgenv-gnu/26.3:v1
#SBATCH --view=default
#SBATCH --time=00:30:00
#SBATCH --partition normal

set -o pipefail

STEPS=200
N=400

export MPICH_GPU_SUPPORT_ENABLED=1

export SPHEXA_MIXD_MAX_LEVEL_DIFF=21
export SPHEXA_SPACIAL_BINS=0
export SPHEXA_WRITE_RANK_COORDS=1

# ENABLE_PROFILE_VALUES: each run is repeated for every value listed here. ENABLE_PROFILE
# controls the per-rank profile.h5 dump and the halo-count analysis on it:
#   0 = disabled (no --profile / -op flags)
#   1 = enabled (print_rank_halo_counts.py runs at the end on all profiled runs)
ENABLE_PROFILE_VALUES=(1)

LOG_FILES=()
PROFILE_DIRS=()

for SPHEXA_BOUNDARY_TYPE in 0; do
    export SPHEXA_BOUNDARY_TYPE

    for ENABLE_PROFILE in "${ENABLE_PROFILE_VALUES[@]}"; do
        if (( ENABLE_PROFILE )); then
            PROFILE_FLAG="_prof"
        else
            PROFILE_FLAG="_noprof"
        fi
        export OUTPUT_FOLDER=windshock_r${SLURM_NTASKS}_c${SLURM_CPUS_PER_TASK}_s${STEPS}_n${N}_gpu_BT${SPHEXA_BOUNDARY_TYPE}_SB${SPHEXA_SPACIAL_BINS}_nonmixd${PROFILE_FLAG}_gpumpi
        mkdir -p $OUTPUT_FOLDER
        export OUTPUT_FILE=${OUTPUT_FOLDER}/output
        rm -f $OUTPUT_FILE
        RUN_LOG=${OUTPUT_FOLDER}/run.log
        LOG_FILES+=("$RUN_LOG")
        PROFILE_ARGS=()
        if (( ENABLE_PROFILE )); then
            PROFILE_DIRS+=("$OUTPUT_FOLDER")
            PROFILE_ARGS=(--profile -op "${OUTPUT_FOLDER}/profile")
        fi
        # --profile makes propagator->writeMetrics() write per-rank numHalos, numParticles, etc.
        # at the end of the run; -op puts it in ${OUTPUT_FOLDER}/profile.h5 instead of ./profile.h5,
        # so concurrent jobs launched from this directory don't overwrite each other's profile.
        # The "Running ... with ... SPHEXA_BOUNDARY_TYPE=" header is what compare_runs_log.py splits runs on.
        echo "Running nonmixd with SPHEXA_BOUNDARY_TYPE=$SPHEXA_BOUNDARY_TYPE SPHEXA_SPACIAL_BINS=$SPHEXA_SPACIAL_BINS SPHEXA_MIXD_MAX_LEVEL_DIFF=$SPHEXA_MIXD_MAX_LEVEL_DIFF ENABLE_PROFILE=$ENABLE_PROFILE" | tee "$RUN_LOG"
        OMP_NUM_THREADS=64 srun --mpi=cray_shasta ./mps_wrapper.sh ./build_gpu_release_nonmixd/main/src/sphexa/sphexa-cuda --init wind-shock -s ${STEPS} -n ${N} --glass 50c.h5 -o $OUTPUT_FILE -w 0 "${PROFILE_ARGS[@]}" 2>&1 | tee -a "$RUN_LOG"
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
