import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import shap
import tensorflow as tf
import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.utils import Sequence, to_categorical
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

from scipy.io import mmread
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


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
df = pd.DataFrame(filtered_matrix_csr.toarray().T, index=filtered_barcodes_data, columns=df_features['name'])

# Load the sparse matrix into a DataFrame
# sparse_df = pd.DataFrame.sparse.from_spmatrix(filtered_matrix_csr)
# df = sparse_df.sparse.to_coo().transpose()
# df = pd.DataFrame.sparse.from_spmatrix(df, index=filtered_barcodes_data, columns=df_features['name'])

# Add a new column 'healthy' to df
df['type'] = annot_data_labels
df['healthy'] = df['type'].str.startswith("SD").astype(int)
df = df.drop('type', axis=1)

# Count the number of 1s and 0s in df['healthy']
num_1 = df['healthy'].sum()
num_0 = len(df['healthy']) - num_1

# Remove some datapoints with label 0
num_samples = min(num_0, num_1)
df_0 = df[df['healthy'] == 0].sample(num_samples, random_state=42)
df_1 = df[df['healthy'] == 1].sample(num_samples, random_state=42)
df_balanced = pd.concat([df_0, df_1])

# Shuffle the dataframe
df_balanced = df_balanced.sample(frac=1, random_state=42)

# Count the number of 1s and 0s in df['healthy']
num_1 = df_balanced['healthy'].sum()
num_0 = len(df_balanced['healthy']) - num_1
# print("Number of 1s:", num_1)
# print("Number of 0s:", num_0)

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
#df = df.head(100).copy()
labels = to_categorical(df_balanced['healthy'], num_classes=2)
dataset = df_balanced.drop('healthy', axis=1)

print("labels")
print(pd.DataFrame(labels).head(100))
print("dataset")
print(dataset.head(100))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, random_state=42)

print("Xtrain")
print(X_train.head(100))
print("ytrain")
print(pd.DataFrame(y_train).head(100))

# Deleting useless variables to free up memory
del df
# del dataset
del labels

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
                    batch_size=X_train.shape[0],
                    validation_split=0.2,
                    callbacks=[early_stopping])

# Plot the accuracy evolution for each epoch
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('./data/plot_accuracy.png')
plt.show()


########################################
########################################
########################################

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy}')
# Predict the labels for the test set
y_pred = model.predict(X_test)

# Convert the predicted probabilities to binary labels
y_pred_binary = np.argmax(y_pred, axis=1)

# Convert the true labels to binary labels
y_test_binary = np.argmax(y_test, axis=1)

# Compute the F1 score
f1 = f1_score(y_test_binary, y_pred_binary)

print("F1 Score:", f1)

X_train_np = X_train.to_numpy()
X_test_np = X_test.to_numpy()
background = X_train_np[np.random.choice(X_train_np.shape[0], 100, replace=False)]

explainer = shap.DeepExplainer(model, background)
shap_values = explainer.shap_values(X_test_np)
shap.summary_plot(shap_values, X_test_np)
# Store the plot in local storage
plt.savefig('./data/plot2.png')
print("Plot saved to local storage")

# Save the model to local storage
model.save('./data/model_raw_2.h5')
print("Model saved to local storage")

""" # Load the model from local storage
loaded_model = keras.models.load_model('/home/thomas/Documents/Imperial/Thesis/Project_repo/data/model_raw.h5')

# Print the number of parameters in the model
num_params = loaded_model.count_params()
print("Number of parameters in the model:", num_params) """
