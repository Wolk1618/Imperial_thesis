import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import shap
import xgboost
import tensorflow as tf

from keras.utils import to_categorical
from scipy.io import mmread
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from numpy import sort
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import json


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


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

# Convert the 'healthy' column to one-hot encoding
labels = to_categorical(df_balanced['healthy'], num_classes=2)
dataset = df_balanced.drop('healthy', axis=1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, random_state=42)

# Converting the training and testing datasets to numpy arrays
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()

# Scaling the training and testing datasets
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


########################################
############### XGBoost ################
########################################

model = xgboost.XGBClassifier(device='cuda', importance_type='gain')
model.fit(X_train_scaled, y_train)


########################################
######## Evaluate the model ############
########################################

# Evaluate the model
accuracy = model.score(X_test_scaled, y_test)
print("Test Accuracy:", accuracy)

# Predict the labels for the test set
y_pred = model.predict(X_test_scaled)

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
model.save_model('./data/model_xgboost.model')
print("Model saved to local storage")

# # Load the model from local storage
# model = xgboost.XGBClassifier()
# model.load_model('./data/model_xgboost.model')

# Compute and display train accuracy
train_accuracy = model.score(X_train_scaled, y_train)
print("Train Accuracy:", train_accuracy)


########################################
############# SHAP PLOTS ###############
########################################

# Extract feature names from the original AnnData object
feature_names = dataset.columns

explainer = shap.Explainer(model, feature_names=feature_names)
shap_values = explainer(X_train_scaled)

# Remove plot_type='bar' to get the default SHAP summary plot

# # Visualize SHAP values
# shap.summary_plot(shap_values[:, :, 0], X_train_scaled, feature_names=feature_names, plot_type='bar', color='#003E74')
# plt.savefig('./data/wdsd_shap_xgboost.png')

# print("Plot saved to local storage")

mean_shap_values = np.abs(shap_values[:, :, 0].values).mean(axis=0)

# # Store SHAP values to a JSON file
# shap_values_dict = dict(zip(feature_names, mean_shap_values))
# shap_values_dict = {k: float(v) for k, v in shap_values_dict.items()}  # Convert float32 values to float
# with open('./data/wdsd_shap_values.json', 'w') as f:
#     json.dump(shap_values_dict, f)

# print("SHAP values saved to JSON file.")

features_1_10 = range(10)
features_1_100 = range(10, 100, 10)
features_1_1000 = range(100, 1100, 100)
top_features = list(features_1_10) + list(features_1_100) + list(features_1_1000)

for i in top_features:
    # Extract the most important features based on SHAP values
    important_features = np.argsort(mean_shap_values)[-i:]
    important_feature_names = feature_names[important_features]

    # Subset the data to include only the most important features
    X_train_important = X_train_scaled[:, important_features]
    X_test_important = X_test_scaled[:, important_features] 

    # Train a new XGBoost classifier using only the 10 most important features
    new_model = xgboost.XGBClassifier()
    new_model.fit(X_train_important, y_train)

    # Evaluate the new model
    y_pred = new_model.predict(X_test_important)
    accuracy = accuracy_score(y_test, y_pred)
    print("n=%d, Accuracy: %.2f%%" % (i, accuracy*100.0))


########################################
########## FEATURE IMPORTANCE ##########
########################################

# model.get_booster().feature_names = list(dataset.columns)

# xgboost.plot_importance(model, max_num_features = 20, color='#003E74', importance_type='weight')
# plt.savefig('./data/wdsd_plot_importance_weight.png')

# # Get feature importances
# feature_importances = model.feature_importances_

# # Convert feature_importances to a regular Python list
# feature_importances_list = feature_importances.tolist()

# # Create a dictionary to store feature importances
# feature_importances_dict = {feature: importance for feature, importance in zip(dataset.columns, feature_importances_list)}

# # Save feature importances to a JSON file
# with open('./data/feature_importances.json', 'w') as f:
#     json.dump(feature_importances_dict, f)

# print("Feature importances saved to JSON file.")

# thresholds = []
# sorted_feature_importances = np.sort(feature_importances)[::-1]

# for i in range(10):
#     thresholds.append(sorted_feature_importances[i])

# for i in range(19, 100, 10):
#     thresholds.append(sorted_feature_importances[i])

# for i in range(199, 1000, 100):
#     thresholds.append(sorted_feature_importances[i])

# print("\nStarting feature selection...\n")

# for thresh in thresholds:
#     # select features using threshold
#     selection = SelectFromModel(model, threshold=thresh, prefit=True)
#     select_X_train = selection.transform(X_train_scaled)
#     # train model
#     selection_model = xgboost.XGBClassifier(device='cuda')
#     selection_model.fit(select_X_train, y_train)
#     # eval model
#     select_X_test = selection.transform(X_test_scaled)
#     predictions = selection_model.predict(select_X_test)
#     accuracy = accuracy_score(y_test, predictions)
#     print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))