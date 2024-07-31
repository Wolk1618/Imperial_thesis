import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import mmread
from scipy.sparse import csr_matrix


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


# Load data
rawData = mmread("./data/cd45+/matrix.mtx")
barcodes = pd.read_csv("./data/cd45+/barcodes.tsv", header=None, names=["barcode"])
features = pd.read_csv("./data/cd45+/features.tsv", sep="\t", header=None, names=["code", "name", "type"])

# Create SingleCellExperiment-like object
adata = sc.AnnData(X=rawData.T, var=features, obs=barcodes)
adata.var_names = features["name"]
adata.obs_names = barcodes["barcode"]

# Get mitochondrial genes
is_mito = adata.var_names.str.startswith("mt-")
adata.var["mito"] = is_mito
#print(sum(is_mito))

# Calculate QC metrics
sc.pp.calculate_qc_metrics(adata, qc_vars=["mito"], percent_top=None, log1p=False, inplace=True)

# Rename column name 0 to orig.ident
adata.obs.rename(columns={"barcode": "orig.ident"}, inplace=True)


# Create metaData matrix
metaData = pd.DataFrame({
    "staticNr": adata.obs_names,
    "orig.ident": adata.obs["orig.ident"],
    "nGene": adata.obs["n_genes_by_counts"],
    "nUMI": adata.obs["total_counts"],
    "percent.mito": adata.obs["pct_counts_mito"]
})
metaData["staticNr"] = 1

listLabels = list(set([int(str(barcode)[-1]) for barcode in barcodes["barcode"]]))

# Add sample names to metaData
for i in listLabels:
    toSearch = f"-{i}"
    metaData.loc[metaData.index.str.contains(toSearch), "orig.ident"] = i

# Get outliers
nmad_low_feature = 3
nmad_high_feature = 3
nmad_low_UMI = 3
nmad_high_UMI = 3
nmad_high_mito = 3

# Calculate outliers
feature_drop_low = is_outlier(adata.obs["n_genes_by_counts"], nmads=nmad_low_feature, type='lower')
feature_drop_high = is_outlier(adata.obs["n_genes_by_counts"], nmads=nmad_high_feature, type='higher')
feature_drop = feature_drop_low | feature_drop_high

libsize_drop_low = is_outlier(adata.obs["total_counts"], nmads=nmad_low_UMI, type='lower')
libsize_drop_high = is_outlier(adata.obs["total_counts"], nmads=nmad_high_UMI, type='higher')
libsize_drop = libsize_drop_low | libsize_drop_high

mito_drop = is_outlier(adata.obs["pct_counts_mito"], nmads=nmad_high_mito, type='higher')

# Add to metaData matrix
metaData["nGene.drop"] = feature_drop
metaData["nUMI.drop"] = libsize_drop
metaData["mito.drop"] = mito_drop
metaData["final.drop"] = feature_drop | libsize_drop | mito_drop

# Remove outliers
adata = adata[~metaData["final.drop"]]

# Filter rawData
rawDataFiltered = rawData.tocsr()[:, ~metaData["final.drop"]]

# Filter barcodes
barcodes.set_index('barcode', inplace=True)
filteredBarcodes = barcodes[~metaData["final.drop"]]
filteredBarcodes = filteredBarcodes.reset_index()

# Create Seurat object
seuratObj = sc.AnnData(X=rawDataFiltered.T, var=features, obs=filteredBarcodes)
seuratObj.var_names = features["name"]
seuratObj.obs_names = filteredBarcodes["barcode"]

# Add % mitochondrial genes
seuratObj.obs["percent.mito"] = adata.obs["pct_counts_mito"][~metaData["final.drop"]]

################
##### HERE #####
################

# Add sample names to orig.ident
for i in range(1, len(barcodes)):
    toSearch = f"-{i}"
    seuratObj.obs.loc[seuratObj.obs_names.str.contains(toSearch), "orig.ident"] = barcodes[0][i]

# Save seuratObj before tricky part
seuratObj.write("work/data/seuratObj.h5ad")

# Load the object
seuratObj = sc.read("work/data/seuratObj.h5ad")

# Normalize SCT
sc.pp.normalize_total(seuratObj, target_sum=1e4)
sc.pp.log1p(seuratObj)

# PCA
sc.pp.pca(seuratObj, n_comps=50)

# Determine statistically significant PCs
sc.pl.pca_variance_ratio(seuratObj, n_pcs=40)
plt.savefig("work/plots/9b_selectPC.png")

# Cluster the cells
dimsToTry = [20]
resToUse = 0.8

for maxPCs in dimsToTry:
    dimsToUse = range(1, maxPCs + 1)
    print(f"Working on 1:{maxPCs}")

    # Find clusters
    sc.pp.neighbors(seuratObj, n_neighbors=10, n_pcs=maxPCs)
    sc.tl.louvain(seuratObj, resolution=resToUse)

    # Create tSNE plot
    sc.tl.tsne(seuratObj, n_pcs=maxPCs)
    sc.pl.tsne(seuratObj, color="louvain", save=f"work/plots/10a_tSNE_{min(dimsToUse)}-{max(dimsToUse)}.png")

    # Create UMAP plot
    sc.tl.umap(seuratObj, n_neighbors=30, n_pcs=maxPCs)
    sc.pl.umap(seuratObj, color="louvain", save=f"work/plots/10b_UMAP_{min(dimsToUse)}-{max(dimsToUse)}.png")

# Save object
seuratObj.write("work/data/seuratObj.h5ad")
