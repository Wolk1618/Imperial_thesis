import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap
import networkx as nx
import community as community_louvain

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import f1_score

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping

from scipy.io import mmread

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


##############################
#### Loading raw dataset #####
##############################

# # Load the matrix.mtx file and convert it to CSR format
# matrix = mmread('./data/cd45+/matrix.mtx')
# matrix_csr = matrix.tocsr()

# # Load the gene names from features.tsv file
# barcodes_data = np.loadtxt('./data/cd45+/barcodes.tsv', dtype=str)

# # Load the cell names from annot_cd45pos.csv file
# annot_data_cell_names = np.loadtxt('./data/cd45+/annot_cd45pos.csv', delimiter=',', usecols=(0), dtype=str)
# # Remove the " at the beginning and end of each string element
# annot_data_cell_names = np.array([s.strip('"') for s in annot_data_cell_names][1:])

# # Filtering the CSR matrix
# filtered_barcodes_data = np.intersect1d(barcodes_data, annot_data_cell_names)
# filtered_barcodes_indexes = [np.where(barcodes_data == barcode)[0][0] for barcode in filtered_barcodes_data]
# filtered_matrix_csr = matrix_csr[:, filtered_barcodes_indexes]

# # Load the features.tsv file to a dataframe
# features_data = np.loadtxt('./data/cd45+/features.tsv', dtype=str, delimiter='\t')
# df_features = pd.DataFrame(features_data, columns=['code', 'name', 'type'])

# # Load the labels from annot_cd45pos.csv file
# annot_data_labels = np.loadtxt('./data/cd45+/annot_cd45pos.csv', delimiter=',', usecols=(4), dtype=str)
# annot_data_labels = [s.strip('"') for s in annot_data_labels][1:]

# # Reorder annot_data_labels to match barcodes_data
# filtered_data_labels_index = [np.where(annot_data_cell_names == barcode)[0][0] for barcode in filtered_barcodes_data]
# annot_data_labels_reordered = [annot_data_labels[i] for i in filtered_data_labels_index]
# annot_data_labels = annot_data_labels_reordered

# # Deleting useless variables
# del barcodes_data
# del features_data
# del matrix
# del matrix_csr
# del annot_data_cell_names
# del annot_data_labels_reordered

# # Create a dense DataFrame from the CSR matrix
# df = pd.DataFrame(filtered_matrix_csr.toarray().T, index=filtered_barcodes_data, columns=df_features['name'])

# # Add a new column 'healthy' to df
# df['type'] = annot_data_labels
# df['healthy'] = df['type'].str.startswith("SD").astype(int)
# df = df.drop('type', axis=1)

# # Count the number of 1s and 0s in df['healthy']
# num_1 = df['healthy'].sum()
# num_0 = len(df['healthy']) - num_1

# # Remove some datapoints with label 0
# num_samples = min(num_0, num_1)
# df_0 = df[df['healthy'] == 0].sample(num_samples, random_state=42)
# df_1 = df[df['healthy'] == 1].sample(num_samples, random_state=42)
# df_balanced = pd.concat([df_0, df_1])

# # Shuffle the dataframe
# df_balanced = df_balanced.sample(frac=1, random_state=42)

# # Convert the 'healthy' column to one-hot encoding
# labels = to_categorical(df_balanced['healthy'], num_classes=2)
# dataset = df_balanced.drop('healthy', axis=1)

# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, random_state=42)

# # Converting the training and testing datasets to numpy arrays
# X_train = X_train.to_numpy()
# X_test = X_test.to_numpy()

# # Scaling the training and testing datasets
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)


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

# # Apply the chosen dimensionality reduction method
method = "UMAP"
dim = 5

if method == "PCA":
    pca_final = PCA(n_components=dim)
    X_train_dr = pca_final.fit_transform(X_train_pca[:, :n_pcs])
    X_test_dr = pca_final.transform(X_test_pca[:, :n_pcs])
    plt.scatter(X_train_dr[:, 0], X_train_dr[:, 1], c=train_labels, cmap='viridis')
    plt.savefig(f"./data/plots/{preprocessing_type}_PCA_{dim}.png")
    plt.close()

if method == "UMAP":
    reducer = umap.UMAP(n_components=dim)
    X_train_dr = reducer.fit_transform(X_train_pca[:, :n_pcs])
    X_test_dr = reducer.transform(X_test_pca[:, :n_pcs])
    plt.scatter(X_train_dr[:, 0], X_train_dr[:, 1], c=train_labels, cmap='viridis')
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