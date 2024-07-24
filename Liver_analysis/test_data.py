import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.io import mmread


# Load the matrix.mtx file
matrix = mmread('./data/cd45+/matrix.mtx')
# Convert the matrix to CSR format
matrix_csr = matrix.tocsr()

#print(matrix_csr.shape)

# Load the barcodes.tsv file
barcodes_data = np.loadtxt('./data/cd45+/barcodes.tsv', dtype=str)
#print(barcodes_data[0])


# Load the cell names from annot_cd45pos.csv file
annot_data_cell_names = np.loadtxt('./data/cd45+/annot_cd45pos.csv', delimiter=',', usecols=(0), dtype=str)
# Remove the " at the beginning and end of each string element
#parsed_annot_data = [[s.strip('"') for s in row] for row in annot_data]
#annot_data = pd.DataFrame(parsed_annot_data[1:], columns=parsed_annot_data[0])
annot_data_cell_names = [s.strip('"') for s in annot_data_cell_names][1:]

filtered_barcodes_data = np.intersect1d(barcodes_data, annot_data_cell_names)
filtered_barcodes_indexes = [np.where(barcodes_data == barcode)[0][0] for barcode in filtered_barcodes_data]
#print(len(filtered_barcodes_indexes))

# Filtering the CSR matrix
filtered_matrix_csr = matrix_csr[:, filtered_barcodes_indexes]
#print(filtered_matrix_csr.shape)

# Deleting useless variables
del barcodes_data
del matrix
del matrix_csr

""" dense_matrix = filtered_matrix_csr.toarray()
print(dense_matrix) """

# Load the features.tsv file
features_data = np.loadtxt('./data/cd45+/features.tsv', dtype=str, delimiter='\t')
# Load features_data into a DataFrame
df_features = pd.DataFrame(features_data, columns=['code', 'name', 'type'])
del features_data
#print(df_features)

# Load the labels from annot_cd45pos.csv file
annot_data_labels = np.loadtxt('./data/cd45+/annot_cd45pos.csv', delimiter=',', usecols=(4), dtype=str)
annot_data_labels = [s.strip('"') for s in annot_data_labels][1:]

# Load the sparse matrix into a DataFrame
#df = pd.DataFrame(dense_matrix, columns=filtered_barcodes_data) # Check that column names are the appropriate ones
sparse_df = pd.DataFrame.sparse.from_spmatrix(filtered_matrix_csr)
sparse_df = pd.DataFrame.sparse.from_spmatrix(sparse_df.sparse.to_coo().transpose(), index=filtered_barcodes_data, columns=df_features['name'])

# Add a new column 'healthy' to df
sparse_df['type'] = annot_data_labels
sparse_df['healthy'] = sparse_df['type'].str.startswith("SD").astype(int)
# Remove the 'type' column from sparse_df
sparse_df = sparse_df.drop('type', axis=1)
print(sparse_df.sample(n=20))


# Check if barcodes_data and annot_data have the same elements
""" # Check if barcodes_data and annot_data have the same elements
if set(barcodes_data) == set(annot_data):
    print("barcodes_data and annot_data have the same elements")
else:
    print("barcodes_data and annot_data do not have the same elements")

print("size of barcodes : " + str(len(set(barcodes_data))))
print("size of annot : " + str(len(set(annot_data))))

if set(annot_data).issubset(set(barcodes_data)):
    print("Annot cell names are included in barcode cell names")
else:
    print("Annot cell names are not included in barcode cell names)")

    print(len(set(annot_data).intersection(set(barcodes_data))))
    not_in_intersection = set(annot_data) - set(barcodes_data)
    print("Elements not in the intersection:", not_in_intersection)

    not_in_intersection_indexes = [annot_data.index(element) for element in not_in_intersection]
    print("Indexes of elements not in the intersection:", not_in_intersection_indexes)

    print(annot_data[0 : not_in_intersection_indexes[0] + 20]) """


