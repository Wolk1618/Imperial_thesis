import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

from scipy.io import mmread
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


########################################
############ LOAD THE DATA #############
########################################

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

# Deleting useless variables to free up memory
del barcodes_data
del features_data
del matrix
del matrix_csr
del annot_data_cell_names
del annot_data_labels_reordered


########################################
############ ADD THE LABELS ############
########################################

# Create a dense DataFrame from the CSR matrix
df = pd.DataFrame(filtered_matrix_csr.toarray().T, index=filtered_barcodes_data, columns=df_features['name'])

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

# Convert the 'healthy' column to one-hot encoding
labels = to_categorical(df_balanced['healthy'], num_classes=2)
dataset = df_balanced.drop('healthy', axis=1)


########################################
########## SPLIT THE DATASET ###########
########################################

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, random_state=42)

# Deleting useless variables to free up memory
del df
# del dataset
del labels


########################################
############# BASE MODEL ###############
########################################

# Define the model
model = Sequential([
    Input(shape=(dataset.shape[1],), sparse=True),
    Dense(512, activation='relu'),
    Dropout(0.5),
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
########## MODEL EVALUATION ############
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

########################################
############# SAVE MODEL ###############
########################################

# Save the model to local storage
model.save('./data/model_raw.h5')
print("Model saved to local storage")