import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import scipy.sparse as sp
from scipy.io import mmread

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.utils import Sequence, to_categorical
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# class DataGenerator(Sequence):
#     def __init__(self, X, y, batch_size):
#         self.X = X
#         self.y = y
#         self.batch_size = batch_size

#     def __len__(self):
#         return int(np.ceil(len(self.X) / float(self.batch_size)))

#     def __getitem__(self, idx):
#         if (idx + 1) * self.batch_size > len(self.X):
#             batch_X = np.asarray(self.X[idx * self.batch_size:])
#             batch_y = np.asarray(self.y[idx * self.batch_size:])
#             return batch_X, batch_y
#         batch_X = np.asarray(self.X[idx * self.batch_size:(idx + 1) * self.batch_size])
#         batch_y = np.asarray(self.y[idx * self.batch_size:(idx + 1) * self.batch_size])
#         return batch_X, batch_y


# Load the matrix.mtx file and convert it to CSR format
matrix = mmread('./data/cd45+/matrix.mtx')
matrix_csr = matrix.tocsr()

# Load the gene names from features.tsv file
barcodes_data = np.loadtxt('./data/cd45+/barcodes.tsv', dtype=str)

# Load the cell names from annot_cd45pos.csv file
annot_data_cell_names = np.loadtxt('./data/cd45+/annot_cd45pos.csv', delimiter=',', usecols=(0), dtype=str)
# Remove the " at the beginning and end of each string element
annot_data_cell_names = np.array([s.strip('"') for s in annot_data_cell_names][1:])

# Filtering the CSR matrix
filtered_barcodes_data = np.intersect1d(barcodes_data, annot_data_cell_names)
filtered_barcodes_indexes = [np.where(barcodes_data == barcode)[0][0] for barcode in filtered_barcodes_data]
filtered_matrix_csr = matrix_csr[:, filtered_barcodes_indexes]

# Load the features.tsv file to a dataframe
features_data = np.loadtxt('./data/cd45+/features.tsv', dtype=str, delimiter='\t')
df_features = pd.DataFrame(features_data, columns=['code', 'name', 'type'])

# Load the labels from annot_cd45pos.csv file
annot_data_labels = np.loadtxt('./data/cd45+/annot_cd45pos.csv', delimiter=',', usecols=(4), dtype=str)
annot_data_labels = [s.strip('"') for s in annot_data_labels][1:]

# Reorder annot_data_labels to match barcodes_data
filtered_data_labels_index = [np.where(annot_data_cell_names == barcode)[0][0] for barcode in filtered_barcodes_data]
annot_data_labels_reordered = [annot_data_labels[i] for i in filtered_data_labels_index]
annot_data_labels = annot_data_labels_reordered

# Deleting useless variables
del barcodes_data
del features_data
del matrix
del matrix_csr
del annot_data_cell_names
del annot_data_labels_reordered

# Create a dense DataFrame from the CSR matrix
df = pd.DataFrame(filtered_matrix_csr.toarray(), index=filtered_barcodes_data, columns=df_features['name'])

# Load the sparse matrix into a DataFrame
# sparse_df = pd.DataFrame.sparse.from_spmatrix(filtered_matrix_csr)
# df = sparse_df.sparse.to_coo().transpose()
# df = pd.DataFrame.sparse.from_spmatrix(df, index=filtered_barcodes_data, columns=df_features['name'])

# Add a new column 'healthy' to df
df['type'] = annot_data_labels
df['healthy'] = df['type'].str.startswith("SD").astype(int)
df = df.drop('type', axis=1)

""" # Create a new sparse DataFrame from the CSR matrix
csr_matrix = pd.DataFrame.sparse.from_spmatrix(df).sparse.to_csr()
df = pd.DataFrame.sparse.from_spmatrix(csr_matrix, index=filtered_barcodes_data, columns=df_features['name'])

# Make the DataFrame sparse
csr_matrix = sp.coo_matrix(df.values)
sparse_df = pd.DataFrame.sparse.from_spmatrix(csr_matrix, index=filtered_barcodes_data, columns=df_features['name']) """

""" # Store sparse_df to local storage
csr_matrix = df.sparse.to_coo().tocsr()
#sp.save_npz('/home/thomas/Documents/Imperial/Thesis/Project_repo/data/sparse_df.npz', csr_matrix)
print("Dataframe saved to local storage")

# Load sparse_df from local storage
sparse_df = pd.read_csv('/home/thomas/Documents/Imperial/Thesis/Project_repo/data/sparse_df.csv') """

# Convert the 'healthy' column to one-hot encoding
df = df.head(100).copy()
labels = to_categorical(df['healthy'], num_classes=2)
dataset = df.drop('healthy', axis=1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, random_state=42)

# Deleting useless variables to free up memory
""" del df
del dataset
del labels """

# Define the generator
#train_generator = DataGenerator(X_train, y_train, batch_size=100)

########################################
#############  BASE MODEL  #############
########################################

""" # Define the model architecture
model = Sequential()
model.add(Dense(256, activation='relu', input_dim=dataset.shape[1]))
model.add(Dropout(0.3))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=5, validation_data=(X_test, y_test)) """


########################################
###### MODEL PROPOSED BY MISTRAL #######
########################################

# Define the model
model = Sequential([
    Input(shape=(dataset.shape[1],), sparse=True),
    # Dense(1024, activation='relu'),
    # Dropout(0.5),
    # Dense(512, activation='relu'),
    # Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Print the model informations
model.summary()

# Train the model
history = model.fit(X_train, y_train,
                    epochs=100,
                    validation_split=0.2,
                    callbacks=[early_stopping])

########################################
########################################
########################################

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy}')

# Save the model to local storage
model.save('./data/model_raw.h5')
print("Model saved to local storage")

""" # Load the model from local storage
loaded_model = keras.models.load_model('/home/thomas/Documents/Imperial/Thesis/Project_repo/data/model_raw.h5')

# Print the number of parameters in the model
num_params = loaded_model.count_params()
print("Number of parameters in the model:", num_params) """
