#!/bin/bash
#PBS -l select=1:ncpus=16:mem=8gb
#PBS -l walltime=20:00:00
#PBS -N m1_cost_sweep
 
module load tools/prod
module load SciPy-bundle/2022.05-foss-2022a

cd $PBS_O_WORKDIR

python3 main_diabetes.py 