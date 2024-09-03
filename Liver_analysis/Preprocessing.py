import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import mmread
from scipy.sparse import csr_matrix
from SCTransform import SCTransform


# Utility function to calculate outliers
def is_outlier(data, nmads=3, type='lower'):
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    if type == 'lower':
        outliers = data < (median - nmads * mad)
    elif type == 'higher':
        outliers = data > (median + nmads * mad)
    else:
        raise ValueError("type must be 'lower' or 'higher'")
    return outliers


###################
#### Load data ####
###################

rawData = mmread("./data/cd45+/matrix.mtx")
barcodes = pd.read_csv("./data/cd45+/barcodes.tsv", header=None, names=["barcode"])
features = pd.read_csv("./data/cd45+/features.tsv", sep="\t", header=None, names=["code", "name", "type"])

# Create SingleCellExperiment-like object
adata = sc.AnnData(X=rawData.T, var=features, obs=barcodes)
adata.var_names = features["code"]
adata.obs_names = barcodes["barcode"]


###################
### Prepare data ##
###################

# Get mitochondrial genes
is_mito = adata.var_names.str.startswith("mt-")
adata.var["mito"] = is_mito
#print(sum(is_mito))

# Rename column name 0 to orig.ident
adata.obs.rename(columns={"barcode": "orig.ident"}, inplace=True)

listLabels = list(set([int(str(barcode)[-1]) for barcode in barcodes["barcode"]]))

# Number of cells and genes
print(f"Number of cells: {adata.n_obs}")
print(f"Number of genes: {adata.n_vars}")


###################
### QC filtering ##
###################
print("Working on QC filtering")

# Calculate QC metrics
sc.pp.calculate_qc_metrics(adata, qc_vars=["mito"], percent_top=None, log1p=False, inplace=True)

# Set outliers parameters
nmad_low_feature = 4
nmad_high_feature = 4
nmad_low_UMI = 4
nmad_high_UMI = 4
nmad_high_mito = 5

# Calculate library size outliers
feature_drop_low = is_outlier(adata.obs["n_genes_by_counts"], nmads=nmad_low_feature, type='lower')
feature_drop_high = is_outlier(adata.obs["n_genes_by_counts"], nmads=nmad_high_feature, type='higher')
feature_drop = feature_drop_low | feature_drop_high

# Calculate library size outliers
libsize_drop_low = is_outlier(adata.obs["total_counts"], nmads=nmad_low_UMI, type='lower')
libsize_drop_high = is_outlier(adata.obs["total_counts"], nmads=nmad_high_UMI, type='higher')
libsize_drop = libsize_drop_low | libsize_drop_high

# Calculate mitochondrial proportion outliers
mito_drop = is_outlier(adata.obs["pct_counts_mito"], nmads=nmad_high_mito, type='higher')

# Combine outliers
final_drop = feature_drop | libsize_drop | mito_drop

# Remove outliers
adata = adata[~final_drop]
rawDataFiltered = rawData.tocsr()[:, ~final_drop]
barcodes.set_index('barcode', inplace=True)
filteredBarcodes = barcodes[~final_drop]
filteredBarcodes = filteredBarcodes.reset_index()


###################
##### Seurat ######
###################

# Create Seurat object
seuratObj = sc.AnnData(X=rawDataFiltered.T, var=features, obs=filteredBarcodes)
seuratObj.var_names = features["code"]
seuratObj.obs_names = filteredBarcodes["barcode"]

# Add % mitochondrial genes
seuratObj.obs["percent.mito"] = adata.obs["pct_counts_mito"][~final_drop]

# Add sample names to seurat object
for i in listLabels:
    toSearch = f"-{i}"
    seuratObj.obs.loc[seuratObj.obs_names.str.contains(toSearch), "orig.ident"] = i

# Number of cells and genes
print(f"Number of cells: {seuratObj.n_obs}")
print(f"Number of genes: {seuratObj.n_vars}")


###################
###### SCT ########
###################
print("Working on SCT")

SCTransform(seuratObj)
print("SCT done")

# Normalize SCT
sc.pp.normalize_total(seuratObj, target_sum=1e4)
sc.pp.log1p(seuratObj)

# Number of cells and genes
print(f"Number of cells: {seuratObj.n_obs}")
print(f"Number of genes: {seuratObj.n_vars}")


##############################
##### Adding the labels ######
##############################
print("Adding the labels")

# Load the cell names from annot_cd45pos.csv file
annot_data_cell_names = np.loadtxt('./data/cd45+/annot_cd45pos.csv', delimiter=',', usecols=(0), dtype=str)
# Remove the " at the beginning and end of each string element
annot_data_cell_names = np.array([s.strip('"') for s in annot_data_cell_names][1:])

# Load the labels from annot_cd45pos.csv file
annot_data_labels = np.loadtxt('./data/cd45+/annot_cd45pos.csv', delimiter=',', usecols=(4), dtype=str)
annot_data_labels = [s.strip('"') for s in annot_data_labels][1:]

# Filtering the CSR matrix
filtered_barcodes_data = np.intersect1d(seuratObj.obs_names, annot_data_cell_names)

print(f"Number of cells in seuratObj: {seuratObj.n_obs}")
print(f"Number of cells in annot_data_cell_names: {len(annot_data_cell_names)}")
print(f"Number of cells in filtered_barcodes_data: {len(filtered_barcodes_data)}")

# Filter the Seurat object
filtered_barcodes_indexes = [np.where(seuratObj.obs_names == barcode)[0][0] for barcode in filtered_barcodes_data]
filtered_seuratObj = seuratObj[filtered_barcodes_indexes, :].copy()

# Reorder annot_data_labels to match barcodes_data
filtered_data_labels_index = [np.where(annot_data_cell_names == barcode)[0][0] for barcode in filtered_barcodes_data]
annot_data_labels_reordered = [annot_data_labels[i] for i in filtered_data_labels_index]
annot_data_labels = annot_data_labels_reordered

# Deleting useless variables
del filtered_data_labels_index
del filtered_barcodes_indexes
del seuratObj
del annot_data_cell_names
del annot_data_labels_reordered
del filtered_barcodes_data

# Create a dense DataFrame from the seurat object
df = pd.DataFrame(filtered_seuratObj.X.toarray(), index=filtered_seuratObj.obs_names, columns=filtered_seuratObj.var_names)

df['healthy'] = annot_data_labels
del filtered_seuratObj
del annot_data_labels

# Add the labels to df
df['label'] = df['healthy'].str.startswith("SD").astype(int)
df = df.drop('healthy', axis=1)

# Balance the dataset with equal number of 1s and 0s
num_1 = df['label'].sum()
num_0 = len(df['label']) - num_1
num_samples = min(num_0, num_1)
df_0 = df[df['label'] == 0].sample(num_samples, random_state=42)
df_1 = df[df['label'] == 1].sample(num_samples, random_state=42)
df_balanced = pd.concat([df_0, df_1])

# Shuffle the dataframe
df_balanced = df_balanced.sample(frac=1, random_state=42)

df_balanced.index = df_balanced['barcode']
df_balanced = df_balanced.drop('barcode', axis=1)


###################
## Storing data ###
###################
print("Storing data")

# Save the balanced dataset
df_balanced.to_csv("./data/preprocessing_osteopontin.csv")

print("Pre processing done")