import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.utils import to_categorical


# Load the annot_cd45pos.csv file
annot_data = np.loadtxt('./data/cd45+/annot_cd45pos.csv', delimiter=',', usecols=(0, 5, 8, 9), dtype=str)
# Remove the " at the beginning and end of each string element
annot_data = [[s.strip('"') for s in row] for row in annot_data]

# Create a pandas DataFrame from the loaded data
df = pd.DataFrame(annot_data[1:], columns=annot_data[0])
# Convert elements of 'umap_1' and 'umap_2' columns to integers
df['umap_1'] = df['umap_1'].astype(float)
df['umap_2'] = df['umap_2'].astype(float)

""" # Check if the 'type' column starts with "SD"
sd_count = df['type'].str.startswith("SD").value_counts()[True]

# Check if the 'type' column starts with "WD"
wd_count = df['type'].str.startswith("WD").value_counts()[True] """

# Map the 'diet' column values to numeric labels
df['label'] = df['diet'].map({
    'SD 12 weeks': 0,
    'WD 12 weeks': 1,
    'SD 36 weeks': 2,
    'WD 36 weeks': 3,
    'SD 24 weeks': 4,
    'WD 24 weeks': 5
})

#print(df.sample(n=20))

dataset = np.array(df[['umap_1', 'umap_2']])
labels = np.array(df['label'])

num_classes = len(np.unique(labels))
y_train_one_hot = to_categorical(labels, num_classes=num_classes)


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(dataset, y_train_one_hot, test_size=0.2, random_state=42)

# Define the model architecture

model = Sequential()
model.add(Dense(128, activation='relu', input_dim=dataset.shape[1]))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print the number of parameters in the model
print("Number of parameters:", model.count_params())

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
