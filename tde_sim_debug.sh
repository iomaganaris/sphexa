#!/usr/bin/env bash
#SBATCH --job-name=tde_sim
#SBATCH -N32
#SBATCH --ntasks-per-node=8
#SBATCH -c36
#SBATCH -A csstaff
#SBATCH --uenv=prgenv-gnu/26.3:v1
#SBATCH --view=default
#SBATCH --time=04:00:00
#SBATCH --partition normal

export SPHEXA_MIXD_MAX_LEVEL_DIFF=21

export MPICH_GPU_SUPPORT_ENABLED=1

# ENABLE_PROFILE controls the per-rank profile.h5 dump and the halo-count analysis on it:
#   0 = disabled (no --profile / -op flags)
#   1 = enabled (print_rank_halo_counts.py runs at the end on the profiled run)
ENABLE_PROFILE=1

# DISABLE_NEIGHBOR_LISTS controls sphexa-cuda's --disable-neighbor-lists flag:
#   0 = neighbor lists enabled (flag omitted)
#   1 = neighbor lists disabled (flag passed)
DISABLE_NEIGHBOR_LISTS=0

PROFILE_DIRS=()

for SPHEXA_BOUNDARY_TYPE in 0 3; do
    export SPHEXA_BOUNDARY_TYPE

    unset SPHEXA_WRITE_RANK_COORDS

    echo "Running with SPHEXA_BOUNDARY_TYPE=$SPHEXA_BOUNDARY_TYPE" # SPHEXA_MIXD_BITS=$SPHEXA_MIXD_BITS"

    # OMP_NUM_THREADS=64 srun --mpi=cray_shasta ./mps_wrapper.sh ./build_gpu/main/src/sphexa/sphexa-cuda --disable-neighbor-lists --init /capstor/scratch/cscs/sebkelle/gb-tde/10B/evo/tde-s166610.h5 --prop std-disk -s 167001 -o $(pwd)/tde-s166610-BT${SPHEXA_BOUNDARY_TYPE}/output
    # OMP_NUM_THREADS=64 srun --mpi=cray_shasta ./mps_wrapper.sh ./build_gpu_release/main/src/sphexa/sphexa-cuda --disable-neighbor-lists --init /capstor/scratch/cscs/sebkelle/gb-tde/10B/evo/tde-s166610.h5 --prop std-disk -s 166612 -o $(pwd)/tde-s166610-BT${SPHEXA_BOUNDARY_TYPE}_mixd/output
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
    OUTPUT_DIRECTORY=$(pwd)/tde-s166610-BT${SPHEXA_BOUNDARY_TYPE}_N${SLURM_NTASKS}${PROFILE_FLAG}${NEIGHBOR_LISTS_FLAG}_gpumpi #_profile_MIXDIFF${SPHEXA_MIXD_MAX_LEVEL_DIFF}
    mkdir -p $OUTPUT_DIRECTORY
    OUTPUT_FILE=${OUTPUT_DIRECTORY}/output.h5
    rm -f $OUTPUT_FILE ${OUTPUT_DIRECTORY}/output_halos.h5
    PROFILE_ARGS=()
    if (( ENABLE_PROFILE )); then
        PROFILE_DIRS+=("$OUTPUT_DIRECTORY")
        PROFILE_ARGS=(--profile -op "$OUTPUT_DIRECTORY/profile")
    fi
    # OMP_NUM_THREADS=64 srun --mpi=cray_shasta ./mps_wrapper.sh ./build_gpu_release/main/src/sphexa/sphexa-cuda --disable-neighbor-lists --init /capstor/scratch/cscs/sebkelle/gb-tde/10B/evo/tde-s166610.h5 --prop std-disk -s 166610 -o $OUTPUT_FILE -w 1 --write-initial --profile -f x,y,z,h --write-halos
    # OMP_NUM_THREADS=64 srun --mpi=cray_shasta ./mps_wrapper.sh ./build_gpu_release/main/src/sphexa/sphexa-cuda "${NEIGHBOR_LISTS_ARGS[@]}" --init /capstor/scratch/cscs/sebkelle/gb-tde/10B/evo/tde-s166610.h5 --prop std-disk -s 166612 "${PROFILE_ARGS[@]}" -o $OUTPUT_FILE
    OMP_NUM_THREADS=32 srun --mpi=cray_shasta --distribution=block:block --cpu-bind=cores ./mps_wrapper.sh ./build_gpu_release/main/src/sphexa/sphexa-cuda "${NEIGHBOR_LISTS_ARGS[@]}" --init /capstor/scratch/cscs/sebkelle/gb-tde/10B/evo/tde-s166610.h5 --prop std-disk -s 166710 "${PROFILE_ARGS[@]}" -o $OUTPUT_FILE
    # OMP_NUM_THREADS=64 srun --mpi=cray_shasta ./mps_wrapper.sh ./build_gpu_release_nonmixd/main/src/sphexa/sphexa-cuda --disable-neighbor-lists --init /capstor/scratch/cscs/sebkelle/gb-tde/10B/evo/tde-s166610.h5 --prop std-disk -s 166710 -o $OUTPUT_FILE
done

if (( ${#PROFILE_DIRS[@]} )); then
    echo
    echo "===== Per-rank halo counts from profile.h5 ====="
    $(pwd)/venv_sphexa_new/bin/python3 ./print_rank_halo_counts.py "${PROFILE_DIRS[@]}"
fi
