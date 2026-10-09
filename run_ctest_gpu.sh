#!/usr/bin/env bash
#SBATCH --job-name=ctest_gpu
#SBATCH -N1
#SBATCH --ntasks-per-node=12
#SBATCH -A csstaff
#SBATCH --uenv=prgenv-gnu/26.3:v1@santis
#SBATCH --view=default
#SBATCH --time=01:00:00
#SBATCH --partition normal

cd build_gpu_release
ctest -v
# ./domain/test/unit/component_units --gtest_filter='Macs.*:Peers*'
