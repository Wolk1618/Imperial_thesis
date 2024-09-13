import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import doubletdetection

from scipy.io import mmread
from scipy.spatial.distance import mahalanobis


# Utility function to calculate outliers
def is_outlier(data, nmads=3, type="lower"):
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    if type == "lower":
        outliers = data < (median - nmads * mad)
    elif type == "higher":
        outliers = data > (median + nmads * mad)
    else:
        raise ValueError("type must be 'lower' or 'higher'")
    return outliers


###################
#### Load data ####
###################
print("Loading data")

rawData = mmread("./data/cd45+/matrix.mtx")
barcodes = pd.read_csv("./data/cd45+/barcodes.tsv", header=None, names=["barcode"])
features = pd.read_csv(
    "./data/cd45+/features.tsv", sep="\t", header=None, names=["code", "name", "type"]
)

# Modify features to have unique names
duplicate_names = features["name"][features["name"].duplicated()]
previous_names = []
for name in duplicate_names:
    if name in previous_names:
        continue
    label_1 = features["name"][features["name"] == name].index[1]
    features.loc[label_1, "name"] = name + "_1"

    if features["name"][features["name"] == name].size > 1:
        label_2 = features["name"][features["name"] == name].index[1]
        features.loc[label_2, "name"] = f"{name}_2"

    previous_names.append(name)

# Create SingleCellExperiment-like object
adata = sc.AnnData(X=rawData.T, var=features, obs=barcodes)
adata.var_names = features["name"]
adata.obs_names = barcodes["barcode"]


###################
### Prepare data ##
###################
print("Preparing data")

# Get mitochondrial genes
is_mito = adata.var_names.str.startswith("mt-")
adata.var["mito"] = is_mito
# print(sum(is_mito))

# Rename column name 0 to orig.ident
adata.obs.rename(columns={"barcode": "orig.ident"}, inplace=True)

listLabels = list(set([int(str(barcode)[-1]) for barcode in barcodes["barcode"]]))


###################
### QC filtering ##
###################
print("Filtering data")

# Calculate QC metrics
sc.pp.calculate_qc_metrics(
    adata, qc_vars=["mito"], percent_top=None, log1p=False, inplace=True
)

# Create metaData matrix
metaData = pd.DataFrame(
    {
        "staticNr": adata.obs_names,
        "orig.ident": adata.obs["orig.ident"],
        "nGene": adata.obs["n_genes_by_counts"],
        "nUMI": adata.obs["total_counts"],
        "percent.mito": adata.obs["pct_counts_mito"],
    }
)
metaData["staticNr"] = 1

# Add sample names to metaData
for i in listLabels:
    toSearch = f"-{i}"
    metaData.loc[metaData.index.str.contains(toSearch), "orig.ident"] = i

# Set outliers parameters
nmad_low_feature = 4
nmad_high_feature = 4
nmad_low_UMI = 4
nmad_high_UMI = 4
nmad_high_mito = 5

# Calculate library size outliers
feature_drop_low = is_outlier(
    adata.obs["n_genes_by_counts"], nmads=nmad_low_feature, type="lower"
)
feature_drop_high = is_outlier(
    adata.obs["n_genes_by_counts"], nmads=nmad_high_feature, type="higher"
)
feature_drop = feature_drop_low | feature_drop_high

# Calculate library size outliers
libsize_drop_low = is_outlier(
    adata.obs["total_counts"], nmads=nmad_low_UMI, type="lower"
)
libsize_drop_high = is_outlier(
    adata.obs["total_counts"], nmads=nmad_high_UMI, type="higher"
)
libsize_drop = libsize_drop_low | libsize_drop_high

# Calculate mitochondrial proportion outliers
mito_drop = is_outlier(
    adata.obs["pct_counts_mito"], nmads=nmad_high_mito, type="higher"
)

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
barcodes.set_index("barcode", inplace=True)
filteredBarcodes = barcodes[~metaData["final.drop"]]
filteredBarcodes = filteredBarcodes.reset_index()


###################
##### Seurat ######
###################

# Create Seurat object
seuratObj = sc.AnnData(X=rawDataFiltered.T, var=features, obs=filteredBarcodes)
seuratObj.var_names = features["name"]
seuratObj.obs_names = filteredBarcodes["barcode"]

# Add % mitochondrial genes
seuratObj.obs["percent_mito"] = adata.obs["pct_counts_mito"][~metaData["final.drop"]]
seuratObj.obs["n_counts"] = adata.obs["total_counts"][~metaData["final.drop"]]

# Add sample names to seurat object
for i in listLabels:
    toSearch = f"-{i}"
    seuratObj.obs.loc[seuratObj.obs_names.str.contains(toSearch), "orig.ident"] = i


seuratObj.X = seuratObj.X.astype(np.float64)

# Number of cells and genes
print(f"Number of cells: {seuratObj.n_obs}")
print(f"Number of genes: {seuratObj.n_vars}")

########################
##### MV Outliers ######
########################
print("Removing mahalanobis outliers")

# Run PCA
sc.tl.pca(seuratObj, n_comps=50)
pca_data = seuratObj.obsm["X_pca"]

# Compute the mean and covariance matrix of the data
mean = np.mean(pca_data, axis=0)
cov = np.cov(pca_data, rowvar=False)

# Compute the Mahalanobis distance for each sample
distances = [mahalanobis(x, mean, np.linalg.inv(cov)) for x in pca_data]
distances = np.array(distances)

median = np.median(distances)
mad = np.median(np.abs(distances - median))

# Print statistics on distances
print("Distance statistics:")
print("Mean:", np.mean(distances))
print("Median:", median)
print("MAD:", mad)
print("Minimum:", np.min(distances))
print("Maximum:", np.max(distances))

# Identify outliers based on the threshold
threshold = median + 3 * mad
outliers = np.where(distances > threshold)[0]

# Subset the AnnData object to remove outlier cells
seuratObj = seuratObj[~seuratObj.obs.index.isin(outliers)]

# Print number of outliers
print("Total number of cells:", len(pca_data))
print("Number of outliers:", len(outliers))

# Number of cells and genes
print(f"Number of cells: {seuratObj.n_obs}")
print(f"Number of genes: {seuratObj.n_vars}")

########################
### Remove bad cells ###
########################
print("Removing bad cells")

# Print distribution of UMI counts
plt.hist(seuratObj.obs["n_counts"], bins=50)
plt.xlabel("UMI counts")
plt.ylabel("Frequency")
plt.title("Distribution of UMI counts")
plt.savefig("./data/plots/UMI_counts.png")
plt.show()

# Remove low quality cells (low UMI counts and high mitochondrial proportion)
seuratObj = seuratObj[seuratObj.obs["n_counts"] > 1000, :]
# Number of cells and genes
print(f"Number of cells: {seuratObj.n_obs}")
print(f"Number of genes: {seuratObj.n_vars}")

# Print distribution of mitochondrial proportion
plt.hist(seuratObj.obs["percent_mito"], bins=50)
plt.xlabel("Mitochondrial proportion")
plt.ylabel("Frequency")
plt.title("Distribution of mitochondrial proportion")
plt.savefig("./data/plots/mito_proportion.png")
plt.show()

seuratObj = seuratObj[seuratObj.obs["percent_mito"] <= 8, :]

# Number of cells and genes
print(f"Number of cells: {seuratObj.n_obs}")
print(f"Number of genes: {seuratObj.n_vars}")

# Remove contaminating cells (potential doublets) using DoubletDetection
clf = doubletdetection.BoostClassifier()
labels = clf.fit(seuratObj.X).predict()
doublet_scores = clf.doublet_score()  # higher means more likely to be doublet
seuratObj.obs["doublet_scores"] = doublet_scores

# Print distribution of doublet scores
plt.hist(seuratObj.obs["doublet_scores"], bins=50)
plt.xlabel("Doublet scores")
plt.ylabel("Frequency")
plt.title("Distribution of doublet scores")
plt.savefig("./data/plots/doublet_scores.png")
plt.show()

seuratObj = seuratObj[seuratObj.obs["doublet_scores"] < 200, :]

# Number of cells and genes
print(f"Number of cells: {seuratObj.n_obs}")
print(f"Number of genes: {seuratObj.n_vars}")

# Remove actively proliferating cells

g2m_genes = [
    "Hmgb2",
    "Cdk1",
    "Nusap1",
    "Ube2c",
    "Birc5",
    "Tpx2",
    "Top2a",
    "Ndc80",
    "Cks2",
    "Nuf2",
    "Cks1b",
    "Mki67",
    "Tmpo",
    "Cenpf",
    "Tacc3",
    "Fam64a",
    "Smc4",
    "Ccnb2",
    "Ckap2l",
    "Ckap2",
    "Aurkb",
    "Bub1",
    "Kif11",
    "Anp32e",
    "Tubb4b",
    "Gtse1",
    "Kif20b",
    "Hjurp",
    "Cdca3",
    "Hn1",
    "Cdc20",
    "Ttk",
    "Cdc25c",
    "Kif2c",
    "Rangap1",
    "Ncapd2",
    "Dlgap5",
    "Cdca2",
    "Cdca8",
    "Ect2",
    "Kif23",
    "Hmmr",
    "Aurka",
    "Psrc1",
    "Anln",
    "Lbr",
    "Ckap5",
    "Cenpe",
    "Ctcf",
    "Nek2",
    "G2e3",
    "Gas2l3",
    "Cbx5",
    "Cenpa",
]
s_genes = [
    "Mcm5",
    "Pcna",
    "Tyms",
    "Fen1",
    "Mcm2",
    "Mcm4",
    "Rrm1",
    "Ung",
    "Gins2",
    "Mcm6",
    "Cdca7",
    "Dtl",
    "Prim1",
    "Uhrf1",
    "Mlf1ip",
    "Hells",
    "Rfc2",
    "Rpa2",
    "Nasp",
    "Rad51ap1",
    "Gmnn",
    "Wdr76",
    "Slbp",
    "Ccne2",
    "Ubr7",
    "Pold3",
    "Msh2",
    "Atad2",
    "Rad51",
    "Rrm2",
    "Cdc45",
    "Cdc6",
    "Exo1",
    "Tipin",
    "Dscc1",
    "Blm",
    "Casp8ap2",
    "Usp1",
    "Clspn",
    "Pola1",
    "Chaf1b",
    "Brip1",
    "E2f8",
]


sc.tl.score_genes_cell_cycle(seuratObj, s_genes=s_genes, g2m_genes=g2m_genes)

# Plot phase distribution
plt.hist(seuratObj.obs["phase"], bins=50)
plt.xlabel("Cell cycle phase")
plt.ylabel("Frequency")
plt.title("Distribution of cell cycle phase")
plt.savefig("./data/plots/cell_cycle_phase.png")
plt.show()

seuratObj = seuratObj[seuratObj.obs["phase"] != "G2M", :]

# Number of cells and genes
print(f"Number of cells: {seuratObj.n_obs}")
print(f"Number of genes: {seuratObj.n_vars}")

################################################
######## Remove low-abundance genes ############
################################################
print("Removing low-abundance genes")

sc.pp.filter_genes(seuratObj, min_cells=3)

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


###################
## Normalisation ##
###################
print("Normalising data")

# Extract the labels and the dataset
labels = df_balanced['label']
dataset = df_balanced.drop('label', axis=1)

# Calculate the total counts per cell
cell_totals = dataset.sum(axis=1)

# Calculate and apply the normalization factors
normalization_factors = 1e4 / cell_totals
normalized_df = dataset.multiply(normalization_factors, axis=0)

# Log-transform the normalized data
dataset = np.log2(normalized_df + 1)

# Add the labels back to the dataset
df_balanced['label'] = labels


###################
#### Store data ###
###################
print("Storing data")

# Save the balanced dataset
df_balanced.to_csv("./data/preprocessing_advanced.csv")

print("Pre processing done")