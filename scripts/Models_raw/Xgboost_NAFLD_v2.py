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

########################################
################ NAFLD #################
########################################

# # Load the matrix.mtx file and convert it to CSR format
# matrix_nafld = mmread('./data/nafld/matrix.mtx')
# matrix_nafld_csr = matrix_nafld.tocsr()

# # Load the gene names from features.tsv file
# barcodes_data_nafld = np.loadtxt('./data/nafld/barcodes.tsv', dtype=str)

# # Load the cell names from annot_cd45pos.csv file
# annot_data_nafld = np.loadtxt('./data/nafld/annot_mouseNafldAll.csv', delimiter=',', usecols=(5, 7), dtype=str)
# # Keep only the samples where the 2nd column is 'scRNAseq'
# annot_data_cell_names_nafld = annot_data_nafld[annot_data_nafld[:, 1] == 'scRnaSeq'][:, 0]
# # Remove the " at the beginning and end of each string element
# # annot_data_cell_names_nafld = np.array([s.strip('"') for s in annot_data_cell_names_nafld][1:])

# # Filtering the CSR matrix
# filtered_barcodes_data_nafld = np.intersect1d(barcodes_data_nafld, annot_data_cell_names_nafld)
# filtered_barcodes_indexes_nafld = [np.where(barcodes_data_nafld == barcode)[0][0] for barcode in filtered_barcodes_data_nafld]
# filtered_matrix_nafld_csr = matrix_nafld_csr[:, filtered_barcodes_indexes_nafld]

# # Load the labels from annot_mouseNafldAll.csv file
# annot_data_labels = np.loadtxt('./data/nafld/annot_mouseNafldAll.csv', delimiter=',', usecols=(8), dtype=str)
# annot_data_labels = [s.strip('"') for s in annot_data_labels][1:]

# # Reorder annot_data_labels to match barcodes_data
# filtered_data_labels_index = [np.where(annot_data_cell_names_nafld == barcode)[0][0] for barcode in filtered_barcodes_data_nafld]
# annot_data_labels_reordered = [annot_data_labels[i] for i in filtered_data_labels_index]
# annot_data_labels = annot_data_labels_reordered

# # Load the features.tsv file to a dataframe
# features_data_nafld = np.loadtxt('./data/nafld/features.tsv', dtype=str, delimiter='\t')
# df_features_nafld = pd.DataFrame(features_data_nafld, columns=['name'])

# # Create a dense DataFrame from the CSR matrix
# df = pd.DataFrame(filtered_matrix_nafld_csr.toarray().T, index=filtered_barcodes_data_nafld, columns=df_features_nafld['name'])

# # Add a new column 'healthy' to df
# df['type'] = annot_data_labels
# df['label'] = df['type'].str.startswith("SD").astype(int)
# df = df.drop('type', axis=1)

# Load the balanced dataset
df = pd.read_csv("./data/preprocessing_basic.csv")


########################################
############# MERGE DATA ###############
########################################

# Count the number of 1s and 0s in df['healthy']
num_1 = df['label'].sum()
num_0 = len(df['label']) - num_1

# Remove some datapoints with label 0
num_samples = min(num_0, num_1)
df_0 = df[df['label'] == 0].sample(num_samples, random_state=42)
df_1 = df[df['label'] == 1].sample(num_samples, random_state=42)
df_balanced = pd.concat([df_0, df_1])

# Shuffle the dataframe
df_balanced = df_balanced.sample(frac=1, random_state=42)

# Convert the labels to one-hot encoding
raw_labels = df_balanced['label']
labels = to_categorical(df_balanced['label'], num_classes=2)
dataset = df_balanced.drop('label', axis=1)
#### TO BE REMOVED
dataset.index = df['barcode']
dataset = dataset.drop('barcode', axis=1)
###

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, random_state=42)

# Converting the training and testing datasets to numpy arrays
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()

# Scaling the training and testing datasets
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ########################################
# ############### XGBoost ################
# ########################################

model = xgboost.XGBClassifier(device='cuda')
model.fit(X_train_scaled, y_train)


# ########################################
# ######### Evaluate the model ###########
# ########################################

# Evaluate the model
accuracy = model.score(X_test_scaled, y_test)
print("Test Accuracy:", accuracy)

# Print train accuracy
y_pred_train = model.predict(X_train_scaled)
train_accuracy = accuracy_score(y_train, y_pred_train)
print("Train Accuracy:", train_accuracy)

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
model.save_model('./data/nafld_alone_model_raw.model')
print("Model saved to local storage")

# # Load the model from local storage
# model = xgboost.XGBClassifier()
# model.load_model('./data/nafld_alone_model_raw.model')


########################################
############# SHAP PLOTS ###############
########################################

# Extract feature names
feature_names = dataset.columns

explainer = shap.Explainer(model, feature_names=feature_names)
shap_values = explainer(X_train_scaled)

# Visualize SHAP values
shap.summary_plot(shap_values[:, :, 0], X_train_scaled, feature_names=feature_names, color='#003E74')
plt.savefig('./data/nafld_alone_shap_xgboost_bees.png')

plt.clf()

shap.summary_plot(shap_values[:, :, 0], X_train_scaled, feature_names=feature_names, plot_type='bar', color='#003E74')
plt.savefig('./data/nafld_alone_shap_xgboost_bar.png')

print("Plots saved to local storage")

mean_shap_values = np.abs(shap_values[:, :, 0].values).mean(axis=0)

# Store SHAP values to a JSON file
shap_values_dict = dict(zip(feature_names, mean_shap_values))
shap_values_dict = {k: float(v) for k, v in shap_values_dict.items()}  # Convert float32 values to float
with open('./data/nafld_alone_shap_values.json', 'w') as f:
    json.dump(shap_values_dict, f)

print("SHAP values saved to JSON file.")


# important_features = np.argsort(mean_shap_values)[-2:]
# important_feature_names = feature_names[important_features]

# print("2 most important features:", important_feature_names)
# print("2 most important features indices:", important_features)

# # Subset the data to include only the most important features
# X_train_important = X_train_scaled[:, important_features]

# X_important_healthy = X_train_important[np.argmax(y_train, axis=1) == 0]
# X_important_nafld = X_train_important[np.argmax(y_train, axis=1) == 1]

# # Plot the reduced dataset
# plt.figure(figsize=(8, 6))
# plt.scatter(X_important_healthy[:, 0], X_important_healthy[:, 1], c='blue', label='Healthy')
# plt.scatter(X_important_nafld[:, 0], X_important_nafld[:, 1], c='red', marker='s', label='Diseased')
# plt.xlabel(important_feature_names[0])
# plt.ylabel(important_feature_names[1])
# plt.xlim(-1, 10)
# plt.ylim(-1, 2)
# plt.title('Dataset Projected onto the 2 Most Important Features')
# plt.legend()
# plt.savefig('./data/nafld_reduced_dataset_plot.png')
# plt.show()

# features_1_10 = range(10)
# features_1_100 = range(10, 100, 10)
# features_1_1000 = range(100, 1000, 100)
# top_features = [features_1_10, features_1_100, features_1_1000]

# for i in top_features:
#     # Extract the most important features based on SHAP values
#     important_features = np.argsort(mean_shap_values)[-i:]
#     important_feature_names = feature_names[important_features]

#     # Subset the data to include only the most important features
#     X_train_important = X_train_scaled[:, important_features]
#     X_test_important = X_test_scaled[:, important_features] 

#     # Train a new XGBoost classifier using only the 10 most important features
#     new_model = xgboost.XGBClassifier()
#     new_model.fit(X_train_important, y_train)

#     # Evaluate the new model
#     y_pred = new_model.predict(X_test_important)
#     accuracy = accuracy_score(y_test, y_pred)
#     print("n=%d, Accuracy: %.2f%%" % (i, accuracy*100.0))


########################################
########## FEATURE IMPORTANCE ##########
########################################

model.get_booster().feature_names = list(dataset.columns)

xgboost.plot_importance(model, max_num_features = 20, color='#003E74')
plt.savefig('./data/nafld_alone_plot_importance.png')

# Get feature importances
feature_importances = model.feature_importances_

# Convert feature_importances to a regular Python list
feature_importances_list = feature_importances.tolist()

# Create a dictionary to store feature importances
feature_importances_dict = {feature: importance for feature, importance in zip(dataset.columns, feature_importances_list)}

# Save feature importances to a JSON file
with open('./data/nafld_alone_feature_importances.json', 'w') as f:
    json.dump(feature_importances_dict, f)

print("Feature importances saved to JSON file.")

# thresholds = []
# sorted_feature_importances = np.sort(feature_importances)[::-1]

# for i in range(10):
#     thresholds.append(sorted_feature_importances[i])

# for i in range(19, 100, 10):
#     thresholds.append(sorted_feature_importances[i])

# for i in range(199, 1000, 100):
#     thresholds.append(sorted_feature_importances[i])


# for thresh in thresholds:
#     # select features using threshold
#     selection = SelectFromModel(model, threshold=thresh, prefit=True)
#     select_X_train = selection.transform(X_train_scaled)
#     # train model
#     selection_model = xgboost.XGBClassifier()
#     selection_model.fit(select_X_train, y_train)
#     # eval model
#     select_X_test = selection.transform(X_test_scaled)
#     predictions = selection_model.predict(select_X_test)
#     accuracy = accuracy_score(y_test, predictions)
#     print("n=%d, Accuracy: %.2f%%" % (select_X_train.shape[1], accuracy*100.0))

# #Identify the indices of the 10 most important features
# important_feature_indices = np.argsort(feature_importances)[-10:]
# important_feature_names = np.array(dataset.columns)[important_feature_indices]

# # Print the 2 most important features
# print("10 most important features:", important_feature_names)
# print("10 most important features indices:", important_feature_indices)

# # Create a reduced dataset containing only the 2 most important features
# X_train_reduced = X_train[:, important_feature_indices]
# X_test_reduced = X_test[:, important_feature_indices]

# # Plot the reduced dataset
# plt.figure(figsize=(8, 6))
# plt.scatter(X_train_reduced[:, 0], X_train_reduced[:, 1], c='red')
# plt.scatter(X_test_reduced[:, 0], X_test_reduced[:, 1], c='blue')
# plt.xlabel(important_feature_names[0])
# plt.ylabel(important_feature_names[1])
# plt.title('Dataset Projected onto the 2 Most Important Features')
# plt.colorbar(label='Class Label')
# plt.savefig('./data/nafld_reduced_dataset_plot.png')
# plt.show()