import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.io import mmread


# # Load the barcodes files
# barcodes_stst = np.loadtxt('./data/barcodes_stst.tsv', dtype=str)
# barcodes_nafld = np.loadtxt('./data/barcodes_nafld.tsv', dtype=str)

# # Print the number of barcodes in each file
# print("Number of barcodes in stst:", len(barcodes_stst))
# print("Number of barcodes in nafld:", len(barcodes_nafld))
# # Print the number of common barcodes
# common_barcodes = np.intersect1d(barcodes_stst, barcodes_nafld)
# print("Number of common barcodes:", len(common_barcodes))

# # Load the features files
# features_stst = np.loadtxt('./data/features_stst.tsv', dtype=str, delimiter='\t')
# features_nafld = np.loadtxt('./data/features_nafld.tsv', dtype=str, delimiter='\t')

# Load the annot files
annot_stst = np.loadtxt('./data/annot_mouseStStAll.csv', delimiter=',', usecols=(5, 7), dtype=str)
annot_nafld = np.loadtxt('./data/annot_mouseNafldAll.csv', delimiter=',', usecols=(5, 7), dtype=str)
annot_cd45pos = np.loadtxt('./data/cd45+/annot_cd45pos.csv', delimiter=',', usecols=(0), dtype=str)
# parsed_annot_stst = [[s.strip('"') for s in row] for row in annot_stst]
parsed_annot_cd45pos = [s.strip('"') for s in annot_cd45pos]

# Extract first row and make it the column names
annot_stst = pd.DataFrame(annot_stst[1:], columns=annot_stst[0])
annot_nafld = pd.DataFrame(annot_nafld[1:], columns=annot_nafld[0])
annot_cd45pos = pd.DataFrame(parsed_annot_cd45pos[1:], columns=['cell'])

# Filter on only the scRnaSeq samples in the column typeSample
annot_stst = annot_stst[annot_stst['typeSample'] == 'scRnaSeq']
annot_nafld = annot_nafld[annot_nafld['typeSample'] == 'scRnaSeq']

print(annot_nafld)
print(annot_cd45pos)

# Compare the number of common cells in the cell names
common_cells = np.intersect1d(annot_cd45pos['cell'], annot_nafld['cell'])
print("Number of common cells:", len(common_cells))

# Print the common cells
print(common_cells)