#!/bin/bash
#PBS -l select=1:ncpus=16:mem=500gb
#PBS -l walltime=5:00:00
#PBS -N xgboost_raw_nafld_shap

module load tools/prod
module load anaconda3/personal
source activate thesis-env

cd $PBS_O_WORKDIR

# python3 ./Liver_analysis/Xgboost_CD45_raw_data.py
python3 ./Liver_analysis/Xgboost_NAFLD_raw_data.py
# python3 ./Liver_analysis/DimReduction_Model.py

# :ngpus=1:gpu_type=RTX6000 dr_nafld_pca2 xgboost_raw_nafld_shap