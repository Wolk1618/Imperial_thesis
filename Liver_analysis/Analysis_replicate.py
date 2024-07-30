import pandas as pd

import matplotlib.pyplot as plt

# Read the CSV file
data = pd.read_csv('./data/annot_humanAll.csv')

# Extract the desired features
features = ['UMAP_1', 'UMAP_2', 'patient']
# Group the data by patient
grouped_data = data.groupby('patient')

# Print the number of unique patients
num_patients = len(grouped_data)
print("Number of unique patients:", num_patients)
""" 
# Create a figure
fig, ax = plt.subplots(figsize=(10, 5))

# Plot the data points for each patient
for patient, patient_data in grouped_data:
    ax.scatter(patient_data['UMAP_1'], patient_data['UMAP_2'], label=patient)

# Set labels and title
ax.set_xlabel('UMAP1')
ax.set_ylabel('UMAP2')
ax.set_title('Patient Analysis')

# Add legend
ax.legend()

# Show the plot
plt.show() """

"""
# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))


# Plot the data points for Obese diet
obese_data = grouped_data.get_group('Obese')
ax1.scatter(obese_data['UMAP_1'], obese_data['UMAP_2'], label='Obese')
ax1.set_xlabel('UMAP1')
ax1.set_ylabel('UMAP2')
ax1.set_title('Obese Diet')

# Plot the data points for Lean diet
lean_data = grouped_data.get_group('Lean')
ax2.scatter(lean_data['UMAP_1'], lean_data['UMAP_2'], label='Lean')
ax2.set_xlabel('UMAP1')
ax2.set_ylabel('UMAP2')
ax2.set_title('Lean Diet')

# Add legend
ax1.legend()
ax2.legend()

# Adjust spacing between subplots
plt.tight_layout()
"""

# Show the plot
plt.show()
