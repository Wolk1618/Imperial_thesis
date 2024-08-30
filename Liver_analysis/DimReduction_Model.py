import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap
import networkx as nx
import community as community_louvain

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout


##############################
##### Splitting dataset ######
##############################

# Load the balanced dataset
df_balanced = pd.read_csv("./data/preprocessing_osteopontin.csv")

# Convert the labels to one-hot encoding
labels = to_categorical(df_balanced['label'], num_classes=2)
dataset = df_balanced.drop('label', axis=1)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, random_state=42)

# Print heads of X_train, X_test, y_train, y_test
print(X_train.head())
print(X_test.head())
print(y_train)
print(y_test)


###################
## Normalisation ##
###################
print("Normalising data")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Log2 Transformation, adding 1 to avoid log(0)
X_train_log2 = np.log2(X_train_scaled + 1)
X_test_log2 = np.log2(X_test_scaled + 1)


# ###################
# ###### PCA ########
# ###################

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


###################
###### PCA ########
###################
print("Working on PCA")

# PCA
pca = PCA(n_components=50)
X_train_pca = pca.fit_transform(X_train_log2)
X_test_pca = pca.transform(X_test_log2)

# Plot PCA variance ratio
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.savefig("./data/plots/selectPC.png")
plt.close()

# Determine the number of statistically significant PCs (e.g., based on elbow method)
n_pcs = 20  # You can choose this based on the plot

###################
## Dim reduction ##
###################
print("Working on Dimensionality Reduction")

# Cluster the cells using Louvain
# Find neighbors
nbrs = NearestNeighbors(n_neighbors=10).fit(X_train_pca[:, :n_pcs])
distances, indices = nbrs.kneighbors(X_train_pca[:, :n_pcs])

# Create a graph
G = nx.Graph()
for i in range(len(indices)):
    for j in indices[i]:
        G.add_edge(i, j, weight=distances[i][j])

# Apply Louvain clustering
partition = community_louvain.best_partition(G, resolution=0.6)
train_labels = pd.Series(partition)

# Apply the chosen dimensionality reduction method
method = "UMAP"
dim = 20

if method == "PCA":
    pca_final = PCA(n_components=dim)
    X_train_dr = pca_final.fit_transform(X_train_pca[:, :n_pcs])
    X_test_dr = pca_final.transform(X_test_pca[:, :n_pcs])
    plt.scatter(X_train_dr[:, 0], X_train_dr[:, 1], c=train_labels, cmap='viridis')
    plt.savefig(f"./data/plots/PCA_{dim}.png")
    plt.close()

if method == "tSNE":
    tsne = TSNE(n_components=dim)
    X_train_dr = tsne.fit_transform(X_train_pca[:, :n_pcs])
    X_test_dr = tsne.transform(X_test_pca[:, :n_pcs])  # Note: tSNE does not support transform
    plt.scatter(X_train_dr[:, 0], X_train_dr[:, 1], c=train_labels, cmap='viridis')
    plt.savefig(f"./data/plots/tSNE_{dim}.png")
    plt.close()

if method == "UMAP":
    reducer = umap.UMAP(n_components=dim)
    X_train_dr = reducer.fit_transform(X_train_pca[:, :n_pcs])
    X_test_dr = reducer.transform(X_test_pca[:, :n_pcs])
    plt.scatter(X_train_dr[:, 0], X_train_dr[:, 1], c=train_labels, cmap='viridis')
    plt.savefig(f"./data/plots/UMAP_{dim}.png")
    plt.close()

# Create a DataFrame with the dimensionality reduction results for training set
columns_name = [f"{method}_{i+1}" for i in range(dim)]
train_dr_df = pd.DataFrame(X_train_dr, columns=columns_name)
train_dr_df["barcode"] = X_train.index
train_dr_df = train_dr_df[["barcode", *columns_name]]

# Create a DataFrame with the dimensionality reduction results for testing set
test_dr_df = pd.DataFrame(X_test_dr, columns=columns_name)
test_dr_df["barcode"] = X_test.index
test_dr_df = test_dr_df[["barcode", *columns_name]]


# Define the model architecture
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=dataset.shape[1]))
model.add(Dense(64, activation='relu', input_dim=dataset.shape[1]))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print the number of parameters in the model
print("Number of parameters:", model.count_params())

# Train the model
#model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))