import pickle
from scipy import stats
import numpy as np
import pandas as pd
from anndata import AnnData
import scipy as sp
import pickle

# Reload full_model_pars and adata from local storage
with open('/home/thomas/Documents/Imperial/Thesis/Project_repo/data/full_model_pars.pkl', 'rb') as f:
    full_model_pars = pickle.load(f)
    
with open('/home/thomas/Documents/Imperial/Thesis/Project_repo/data/adata.pkl', 'rb') as f:
    adata = pickle.load(f)


print("size of adata : ", adata)
print("size of full_model_pars : ", full_model_pars.shape)

print("full_model_pars : ", full_model_pars)
print("full_model_pars.columns : ", full_model_pars.columns)
print("adata.var : ", adata.var)
print("adata.var.columns : ", adata.var.columns)
print("adata.obs : ", adata.obs)
print("adata.obs.columns : ", adata.obs.columns)


extended_model_pars = pd.DataFrame(data = np.zeros((adata.var.shape[0], full_model_pars.shape[1])), index=adata.var.index, columns=full_model_pars.columns)
full_model_pars = full_model_pars.reindex(extended_model_pars.index)
extended_model_pars.loc[full_model_pars.index, full_model_pars.columns] = full_model_pars.values
full_model_pars = extended_model_pars