import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Load Seurat object
seuratObj = sc.read("./data/OstPreprocessingSeuratObj.h5ad")

# Number of cells and genes
print(f"Number of cells: {seuratObj.n_obs}")
print(f"Number of genes: {seuratObj.n_vars}")

# ###################
# ## Normalisation ##
# ###################
# print("Normalising data")

# sc.pp.normalize_total(seuratObj, target_sum=1e4)
# sc.pp.log1p(seuratObj) # !!! use of log1p instead of log2
# # seuratObj.X = np.log2(seuratObj.X + 1)


# # ###################
# # ###### PCA ########
# # ###################

# sc.pp.pca(seuratObj, n_comps=50)

# # Determine statistically significant PCs
# sc.pl.pca_variance_ratio(seuratObj, n_pcs=40)
# plt.savefig("./data/plots/selectPC.png")


# ###################
# ## Dim reduction ##
# ###################
# print("Working on Dimentionality Reduction")

# # Cluster the cells
# dim = 20
# resToUse = 0.6
# method = "UMAP"

# # Find clusters
# sc.pp.neighbors(seuratObj, n_neighbors=10, n_pcs=4)
# sc.tl.louvain(seuratObj, resolution=resToUse)

# if method == "PCA":
#     sc.tl.pca(seuratObj, n_comps=dim)
#     sc.pl.pca(seuratObj, color="louvain", save=f"/PCA_{dim}.png")
#     dr_coords = seuratObj.obsm["X_pca"]

# if method == "tSNE":
#     sc.tl.tsne(seuratObj, n_pcs=dim)
#     sc.pl.tsne(seuratObj, color="louvain", save=f"/tSNE_{dim}.png")
#     dr_coords = seuratObj.obsm["X_tsne"]

# if method == "UMAP":
#     sc.tl.umap(seuratObj, n_components=dim)
#     sc.pl.umap(seuratObj, color="louvain", save=f"/UMAP_{dim}.png")
#     dr_coords = seuratObj.obsm["X_umap"]

# columns_name = [f"{method}_{i+1}" for i in range(dim)]
# dr_df = pd.DataFrame(dr_coords[:, :dim], columns=columns_name)
# dr_df["barcode"] = seuratObj.obs_names
# # dr_df["percent.mito"] = seuratObj.obs["percent.mito"]
# dr_df = dr_df[["barcode", *columns_name]]
# dr_df.to_excel(f"./data/UMAP_{dim}.xlsx", index=False)
