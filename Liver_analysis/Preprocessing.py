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


###################
### QC filtering ##
###################

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


###################
###### SCT ########
###################

SCTransform(seuratObj)
print("SCT done")

# Normalize SCT
sc.pp.normalize_total(seuratObj, target_sum=1e4)
sc.pp.log1p(seuratObj)


###################
###### PCA ########
###################

sc.pp.pca(seuratObj, n_comps=50)

# Determine statistically significant PCs
sc.pl.pca_variance_ratio(seuratObj, n_pcs=40)
plt.savefig("./data/plots/9b_selectPC.png")


###################
###### UMAP #######
###################

# Cluster the cells
dim = 20
resToUse = 0.8

print(f"Working on UMAP")

# Find clusters
sc.pp.neighbors(seuratObj, n_neighbors=10, n_pcs=dim)
sc.tl.louvain(seuratObj, resolution=resToUse)

# Create tSNE plot
sc.tl.tsne(seuratObj, n_pcs=dim)
sc.pl.tsne(seuratObj, color="louvain", save=f"/10a_tSNE_{1}-{dim}.png")

# Create UMAP plot
sc.tl.umap(seuratObj, n_components=dim)
sc.pl.umap(seuratObj, color="louvain", save=f"/10b_UMAP_{1}-{dim}.png")


###################
## Storing data ###
###################

# Save UMAP coordinates to Excel file
umap_coords = seuratObj.obsm["X_umap"]
umap_df = pd.DataFrame(umap_coords[:, :2], columns=["UMAP1", "UMAP2"])
umap_df["barcode"] = seuratObj.obs_names
umap_df["percent.mito"] = seuratObj.obs["percent.mito"]
umap_df = umap_df[["barcode", "percent.mito", "UMAP1", "UMAP2"]]
umap_df.to_excel("./data/umap_coords.xlsx", index=False)
