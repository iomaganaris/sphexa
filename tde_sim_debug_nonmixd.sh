#!/usr/bin/env bash
#SBATCH --job-name=tde_sim_nonmixd
#SBATCH -N64
#SBATCH --ntasks-per-node=4
#SBATCH -c72
#SBATCH -A csstaff
#SBATCH --uenv=prgenv-gnu/26.3:v1
#SBATCH --view=default
#SBATCH --time=04:00:00
#SBATCH --partition normal

export SPHEXA_MIXD_MAX_LEVEL_DIFF=21

export MPICH_GPU_SUPPORT_ENABLED=1

for SPHEXA_BOUNDARY_TYPE in 0; do
    export SPHEXA_BOUNDARY_TYPE

    unset SPHEXA_WRITE_RANK_COORDS

    echo "Running TDE SIM nonMixD with SPHEXA_BOUNDARY_TYPE=$SPHEXA_BOUNDARY_TYPE" # SPHEXA_MIXD_BITS=$SPHEXA_MIXD_BITS"

    # OMP_NUM_THREADS=64 srun --mpi=cray_shasta ./mps_wrapper.sh ./build_gpu/main/src/sphexa/sphexa-cuda --disable-neighbor-lists --init /capstor/scratch/cscs/sebkelle/gb-tde/10B/evo/tde-s166610.h5 --prop std-disk -s 167001 -o $(pwd)/tde-s166610-BT${SPHEXA_BOUNDARY_TYPE}/output
    # OMP_NUM_THREADS=64 srun --mpi=cray_shasta ./mps_wrapper.sh ./build_gpu_release/main/src/sphexa/sphexa-cuda --disable-neighbor-lists --init /capstor/scratch/cscs/sebkelle/gb-tde/10B/evo/tde-s166610.h5 --prop std-disk -s 166612 -o $(pwd)/tde-s166610-BT${SPHEXA_BOUNDARY_TYPE}_mixd/output
    OUTPUT_DIRECTORY=$(pwd)/tde-s166610-BT${SPHEXA_BOUNDARY_TYPE}_N${SLURM_NTASKS}_nonmixd_gpumpi #_profile_MIXDIFF${SPHEXA_MIXD_MAX_LEVEL_DIFF}
    mkdir -p $OUTPUT_DIRECTORY
    OUTPUT_FILE=${OUTPUT_DIRECTORY}/output.h5
    rm -f $OUTPUT_FILE ${OUTPUT_DIRECTORY}/output_halos.h5
    # OMP_NUM_THREADS=64 srun --mpi=cray_shasta ./mps_wrapper.sh ./build_gpu_release/main/src/sphexa/sphexa-cuda --disable-neighbor-lists --init /capstor/scratch/cscs/sebkelle/gb-tde/10B/evo/tde-s166610.h5 --prop std-disk -s 166610 -o $OUTPUT_FILE -w 1 --write-initial --profile -f x,y,z,h --write-halos
    OMP_NUM_THREADS=64 srun --mpi=cray_shasta ./mps_wrapper.sh ./build_gpu_release_nonmixd/main/src/sphexa/sphexa-cuda --init /capstor/scratch/cscs/sebkelle/gb-tde/10B/evo/tde-s166610.h5 --disable-neighbor-lists --prop std-disk -s 166710 -o $OUTPUT_FILE
    # OMP_NUM_THREADS=64 srun --mpi=cray_shasta ./mps_wrapper.sh ./build_gpu_release_nonmixd/main/src/sphexa/sphexa-cuda --disable-neighbor-lists --init /capstor/scratch/cscs/sebkelle/gb-tde/10B/evo/tde-s166610.h5 --prop std-disk -s 166710 -o $OUTPUT_FILE
done

