import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import scipy.sparse as sp
from keras.layers import Dropout
from scipy.io import mmread
from keras.utils import Sequence
import keras


""" class DataGenerator(Sequence):
    def __init__(self, X, y, batch_size):
        self.X = X
        self.y = y
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.X) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_X = np.asarray(self.X[idx * self.batch_size:(idx + 1) * self.batch_size])
        batch_y = np.asarray(self.y[idx * self.batch_size:(idx + 1) * self.batch_size])
        return batch_X, batch_y


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
df = sparse_df.sparse.to_coo().transpose()
df = pd.DataFrame.sparse.from_spmatrix(df, index=filtered_barcodes_data, columns=df_features['name'])

# Add a new column 'healthy' to df
df['type'] = annot_data_labels
df['healthy'] = df['type'].str.startswith("SD").astype(int)
# Remove the 'type' column from sparse_df
df = df.drop('type', axis=1) """
#print(sparse_df.sample(n=20))
""" 
# Convert df to a CSR matrix
csr_matrix = pd.DataFrame.sparse.from_spmatrix(df).sparse.to_csr()

# Create a new sparse DataFrame from the CSR matrix
df = pd.DataFrame.sparse.from_spmatrix(csr_matrix, index=filtered_barcodes_data, columns=df_features['name'])

# Make the DataFrame sparse
csr_matrix = sp.coo_matrix(df.values)
sparse_df = pd.DataFrame.sparse.from_spmatrix(csr_matrix, index=filtered_barcodes_data, columns=df_features['name'])

# Store sparse_df to local storage
csr_matrix = df.sparse.to_coo().tocsr()
#sp.save_npz('/home/thomas/Documents/Imperial/Thesis/Project_repo/data/sparse_df.npz', csr_matrix)
print("Dataframe saved to local storage")


# Load sparse_df from local storage
sparse_df = pd.read_csv('/home/thomas/Documents/Imperial/Thesis/Project_repo/data/sparse_df.csv')
 """
""" labels = df['healthy']
dataset = df.drop('healthy', axis=1)

del sparse_df
del df

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

# Define the generator
train_generator = DataGenerator(X_train, y_train, batch_size=64)

print("Before training the model")
del dataset
del labels

# Train the model
model.fit(train_generator, epochs=5, validation_data=(X_test, y_test))

# Save the model to local storage
model.save('/home/thomas/Documents/Imperial/Thesis/Project_repo/data/model_raw.h5')
print("Model saved to local storage") """

# Load the model from local storage
loaded_model = keras.models.load_model('/home/thomas/Documents/Imperial/Thesis/Project_repo/data/model_raw.h5')

# Print the number of parameters in the model
num_params = loaded_model.count_params()
print("Number of parameters in the model:", num_params)
