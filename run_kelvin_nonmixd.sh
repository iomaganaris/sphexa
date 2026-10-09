#!/usr/bin/env bash
#SBATCH --job-name=kelvin
#SBATCH -N4
#SBATCH --ntasks-per-node=4
#SBATCH -c72
#SBATCH -A csstaff
#SBATCH --uenv=prgenv-gnu/26.3:v1
#SBATCH --view=default
#SBATCH --time=00:20:00
#SBATCH --partition normal

set -o pipefail

STEPS=1
N=50

export SPHEXA_MIXD_MAX_LEVEL_DIFF=21
export SPHEXA_SPACIAL_BINS=0
export SPHEXA_WRITE_RANK_COORDS=1

LOG_FILES=()
PROFILE_DIRS=()

for SPHEXA_BOUNDARY_TYPE in 0 3; do
    export SPHEXA_BOUNDARY_TYPE
    export OUTPUT_FOLDER=kelvin_r${SLURM_NTASKS}_c${SLURM_CPUS_PER_TASK}_s${STEPS}_n${N}_gpu_BT${SPHEXA_BOUNDARY_TYPE}_SB${SPHEXA_SPACIAL_BINS}_nonmixd_profile
    mkdir -p $OUTPUT_FOLDER
    export OUTPUT_FILE=${OUTPUT_FOLDER}/output
    rm -f $OUTPUT_FILE
    RUN_LOG=${OUTPUT_FOLDER}/run.log
    LOG_FILES+=("$RUN_LOG")
    PROFILE_DIRS+=("$OUTPUT_FOLDER")
    # --profile makes propagator->writeMetrics() write per-rank numHalos, numParticles, etc.
    # at the end of the run; -op puts it in ${OUTPUT_FOLDER}/profile.h5 instead of ./profile.h5,
    # so concurrent jobs launched from this directory don't overwrite each other's profile.
    # The "Running ... with ... SPHEXA_BOUNDARY_TYPE=" header is what compare_runs_log.py splits runs on.
    echo "Running nonmixd with SPHEXA_BOUNDARY_TYPE=$SPHEXA_BOUNDARY_TYPE SPHEXA_SPACIAL_BINS=$SPHEXA_SPACIAL_BINS SPHEXA_MIXD_MAX_LEVEL_DIFF=$SPHEXA_MIXD_MAX_LEVEL_DIFF" | tee "$RUN_LOG"
    OMP_NUM_THREADS=64 srun --mpi=cray_shasta ./mps_wrapper.sh ./build_gpu_release_nonmixd/main/src/sphexa/sphexa-cuda --init kelvin-helmholtz -s ${STEPS} -n ${N} --glass 50c.h5 -o $OUTPUT_FILE -w 0 --profile -op ${OUTPUT_FOLDER}/profile 2>&1 | tee -a "$RUN_LOG"
done

echo
echo "===== Last 9 lines of each run ====="
for RUN_LOG in "${LOG_FILES[@]}"; do
    echo
    echo "--- $RUN_LOG ---"
    tail -n 9 "$RUN_LOG"
done

echo
echo "===== Per-rank halo counts from profile.h5 ====="
$(pwd)/venv_sphexa_new/bin/python3 ./print_rank_halo_counts.py "${PROFILE_DIRS[@]}"
