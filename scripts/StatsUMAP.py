import pickle
from scipy import stats
import numpy as np
import pandas as pd
from anndata import AnnData
import scipy as sp


# Load the annot_cd45pos.csv file
annot = pd.read_csv('./data/cd45+/annot_cd45pos.csv', usecols=['cell', 'umap_1', 'umap_2'])

# Load the umap_coords.xlsx file
preproc = pd.read_excel('./data/umap_coords.xlsx', usecols=['barcode', 'UMAP1', 'UMAP2'])

common_items = len(set(annot['cell']).intersection(set(preproc['barcode'])))
print("Number of annotated cells:", len(annot))
print("Number of preprocessed cells:", len(preproc))
print("Number of common items:", common_items)

filtered_annot = annot[annot['cell'].isin(preproc['barcode'])].reset_index(drop=True)
filtered_preproc = preproc[preproc['barcode'].isin(annot['cell'])].reset_index(drop=True)

# Calculate the correlation between the UMAP coordinates
mse = np.mean((filtered_annot['umap_1'] - filtered_preproc['UMAP1'])**2)
print("Mean Squared Error between UMAP1 in annot_cd45pos.csv and UMAP1 in umap_coords.xlsx:", mse)

mse_umap2 = np.mean((filtered_annot['umap_2'] - filtered_preproc['UMAP2'])**2)
print("Mean Squared Error between UMAP2 in annot_cd45pos.csv and UMAP2 in umap_coords.xlsx:", mse_umap2)
