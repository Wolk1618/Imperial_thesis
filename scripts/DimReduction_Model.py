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
from sklearn.metrics import f1_score

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping

##############################
###### Loading dataset #######
##############################

preprocessing_type = "basic"
dataset_path = f"./data/preprocessing_{preprocessing_type}.csv"

# Load the balanced dataset
df_balanced = pd.read_csv(dataset_path)

# Print number of samples
print("Number of samples:", len(df_balanced))
exit()

# Convert the labels to one-hot encoding
labels = to_categorical(df_balanced['label'], num_classes=2)
dataset = df_balanced.drop('label', axis=1)
dataset.index = df_balanced['barcode']
dataset = dataset.drop('barcode', axis=1)


##############################
##### Splitting dataset ######
##############################
print("Splitting dataset")

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, random_state=42)

# # Print heads of X_train, X_test, y_train, y_test
# print(X_train.head())
# print(X_test.head())
# print(y_train)
# print(y_test)


###################
###### PCA ########
###################
print("Working on PCA")

# PCA
pca = PCA(n_components=50)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Plot PCA variance ratio
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.savefig(f"./data/plots/{preprocessing_type}_selectPC.png")
plt.close()

# Determine the number of statistically significant PCs (e.g., based on elbow method)
n_pcs = 10  # You can choose this based on the plot

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
    for j in range(len(indices[i])):
        G.add_edge(i, indices[i][j], weight=distances[i][j])

# Apply Louvain clustering
partition = community_louvain.best_partition(G, resolution=0.6)
train_labels = pd.Series(partition)

# Apply the chosen dimensionality reduction method
method = "PCA"
dim = 20

if method == "PCA":
    pca_final = PCA(n_components=dim)
    X_train_dr = pca_final.fit_transform(X_train_pca[:, :n_pcs])
    X_test_dr = pca_final.transform(X_test_pca[:, :n_pcs])
    plt.scatter(X_train_dr[:, 0], X_train_dr[:, 1], c=train_labels, cmap='viridis')
    # X_train_dr = pca_final.fit_transform(X_train)
    # X_test_dr = pca_final.transform(X_test)
    # plt.scatter(X_train_dr[:, 0], X_train_dr[:, 1], cmap='viridis')
    plt.savefig(f"./data/plots/{preprocessing_type}_PCA_{dim}.png")
    plt.close()

# if method == "tSNE":
#     tsne = TSNE(n_components=dim)
#     X_train_dr = tsne.fit_transform(X_train_pca[:, :n_pcs])
#     X_test_dr = tsne.transform(X_test_pca[:, :n_pcs])  # Note: tSNE does not support transform
#     plt.scatter(X_train_dr[:, 0], X_train_dr[:, 1], c=train_labels, cmap='viridis')
#     plt.savefig(f"./data/plots/{preprocessing_type}_tSNE_{dim}.png")
#     plt.close()

if method == "UMAP":
    reducer = umap.UMAP(n_components=dim)
    X_train_dr = reducer.fit_transform(X_train_pca[:, :n_pcs])
    X_test_dr = reducer.transform(X_test_pca[:, :n_pcs])
    plt.scatter(X_train_dr[:, 0], X_train_dr[:, 1], c=train_labels, cmap='viridis')
    # X_train_dr = reducer.fit_transform(X_train)
    # X_test_dr = reducer.transform(X_test)
    # plt.scatter(X_train_dr[:, 0], X_train_dr[:, 1], cmap='viridis')
    plt.savefig(f"./data/plots/{preprocessing_type}_UMAP_{dim}.png")
    plt.close()


##############################
###### Model definition ######
##############################

# Define the model architecture
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=dim))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print the model informations
model.summary()

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(X_train_dr, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Plot the accuracy evolution for each epoch
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig(f'./data/plots/{preprocessing_type}_plot_accuracy_{method}_{dim}.png')
plt.show()


########################################
####### Evaluate and save model ########
########################################

# Evaluate the model
loss, accuracy = model.evaluate(X_test_dr, y_test)
print(f'Test Accuracy: {accuracy}')
# Predict the labels for the test set
y_pred = model.predict(X_test_dr)

# Convert the predicted probabilities to binary labels
y_pred_binary = np.argmax(y_pred, axis=1)

# Convert the true labels to binary labels
y_test_binary = np.argmax(y_test, axis=1)

# Compute the F1 score
f1 = f1_score(y_test_binary, y_pred_binary)

print("F1 Score:", f1)

# Save the model to local storage
model.save(f'./data/model_{preprocessing_type}_{method}_{dim}.h5')
print("Model saved to local storage")