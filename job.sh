#!/bin/bash
#PBS -l select=1:ncpus=16:mem=500gb
#PBS -l walltime=5:00:00
#PBS -N xgboost_wdsd_shap_reduced

module load tools/prod
module load anaconda3/personal
source activate thesis-env

cd $PBS_O_WORKDIR

python3 ./scripts/Xgboost_CD45_raw_data.py
# python3 ./scripts/Xgboost_NAFLD_raw_data.py
# python3 ./scripts/Xgboost_NAFLD_v2.py
# python3 ./scripts/DimReduction_Model.py

# :ngpus=1:gpu_type=RTX6000 dr_nafld_pca2 xgboost_raw_test_1feature xgboost_wdsd_cover no_dr_preproc_basic xgboost_wdsd_shap_reduced