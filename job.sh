#!/bin/bash
#PBS -l select=1:ncpus=16:mem=32gb:ngpus=1:gpu_type=RTX6000
#PBS -l walltime=1:00:00
#PBS -N raw_rnaseq_nn_gpu
 
module load tools/prod
module load anaconda3/personal
source activate thesis-env

cd $PBS_O_WORKDIR

python3 ./Liver_analysis/ML_CD45_raw_data.py