import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from keras.layers import Dropout
from scipy.io import mmread


# Load the matrix.mtx file
matrix = mmread('./data/cd45+/matrix.mtx')
# Convert the matrix to CSR format
matrix_csr = matrix.tocsr()

# Load the barcodes.tsv file
barcodes_data = np.loadtxt('./data/cd45+/barcodes.tsv', dtype=str)

# Load the cell names from annot_cd45pos.csv file
annot_data_cell_names = np.loadtxt('./data/cd45+/annot_cd45pos.csv', delimiter=',', usecols=(0), dtype=str)
# Remove the " at the beginning and end of each string element
annot_data_cell_names = [s.strip('"') for s in annot_data_cell_names][1:]

# Filtering the CSR matrix
filtered_barcodes_data = np.intersect1d(barcodes_data, annot_data_cell_names)
filtered_barcodes_indexes = [np.where(barcodes_data == barcode)[0][0] for barcode in filtered_barcodes_data]
filtered_matrix_csr = matrix_csr[:, filtered_barcodes_indexes]

# Deleting useless variables
del barcodes_data
del matrix
del matrix_csr

# Load the features.tsv file to a dataframe
features_data = np.loadtxt('./data/cd45+/features.tsv', dtype=str, delimiter='\t')
df_features = pd.DataFrame(features_data, columns=['code', 'name', 'type'])
del features_data

# Load the labels from annot_cd45pos.csv file
annot_data_labels = np.loadtxt('./data/cd45+/annot_cd45pos.csv', delimiter=',', usecols=(4), dtype=str)
annot_data_labels = [s.strip('"') for s in annot_data_labels][1:]

# Load the sparse matrix into a DataFrame
sparse_df = pd.DataFrame.sparse.from_spmatrix(filtered_matrix_csr)
sparse_df = pd.DataFrame.sparse.from_spmatrix(sparse_df.sparse.to_coo().transpose(), index=filtered_barcodes_data, columns=df_features['name'])

# Add a new column 'healthy' to df
sparse_df['type'] = annot_data_labels
sparse_df['healthy'] = sparse_df['type'].str.startswith("SD").astype(int)
# Remove the 'type' column from sparse_df
sparse_df = sparse_df.drop('type', axis=1)
#print(sparse_df.sample(n=20))

dataset = sparse_df.drop('healthy', axis=1)
labels = sparse_df['healthy']

print("Before splitting the dataset")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, random_state=42)

print("After splitting the dataset")

# Define the model architecture
model = Sequential()
model.add(Dense(256, activation='relu', input_dim=dataset.shape[1]))
model.add(Dropout(0.3))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print("Before training the model")

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
