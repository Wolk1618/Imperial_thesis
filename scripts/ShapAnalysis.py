import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from scipy.io import mmread

import keras
from keras.utils import to_categorical
import shap
import matplotlib.pyplot as plt



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
df = pd.DataFrame(filtered_matrix_csr.toarray().T, index=filtered_barcodes_data, columns=df_features['name'])

# Add a new column 'healthy' to df
df['type'] = annot_data_labels
df['healthy'] = df['type'].str.startswith("SD").astype(int)
df = df.drop('type', axis=1)

# Convert the 'healthy' column to one-hot encoding
labels = to_categorical(df['healthy'], num_classes=2)
dataset = df.drop('healthy', axis=1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, random_state=42)

# Deleting useless variables to free up memory
del df
# del dataset
del labels

# Load the model from local storage
loaded_model = keras.models.load_model('./data/model_raw.h5')

X_train_np = X_train.to_numpy()
X_test_np = X_test.to_numpy()
background = X_train_np[np.random.choice(X_train_np.shape[0], 100, replace=False)]

explainer = shap.DeepExplainer(loaded_model, background)
shap_values = explainer.shap_values(X_test_np)
shap.summary_plot(shap_values, X_test_np)
# Store the plot in local storage
plt.savefig('./data/plot.png')
print("Plot saved to local storage")