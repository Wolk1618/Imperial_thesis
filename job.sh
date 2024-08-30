#!/bin/bash
#PBS -l select=1:ncpus=16:mem=160gb
#PBS -l walltime=1:00:00
#PBS -N raw_xgboost_shap
 
module load tools/prod
module load anaconda3/personal
source activate thesis-env

cd $PBS_O_WORKDIR

# python3 ./Liver_analysis/Xgboost_CD45_raw_data.py
python3 ./Liver_analysis/Preprocessing_Osteopontin.py

# :ngpus=1:gpu_type=RTX6000