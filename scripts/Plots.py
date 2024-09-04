import matplotlib.pyplot as plt

# features_number = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
# accuracy = [94.90, 96.05, 96.07, 96.37, 96.43, 96.34, 96.51, 96.41, 96.30, 96.28]

# features_number = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# accuracy = [82.41, 87.48, 89.70, 90.59, 92.32, 93.36, 93.75, 93.88, 95.04, 94.90]
features_number = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
accuracy = [81.24, 84.32, 86.46, 87.21, 87.49, 88.18, 88.50, 89.25, 90.14, 90.71, 94.30, 95.75, 96.23, 96.50, 96.58, 96.53, 96.77, 96.80, 97.03, 97.44, 97.53, 97.57, 97.49, 97.49, 97.48, 97.57, 97.55, 97.51]

plt.plot(features_number, accuracy, color='#003E74')
plt.title('Model Accuracy by Number of Features')
plt.xlabel('Number of Features')
plt.ylabel('Accuracy')
plt.xscale('log')  # Set x-axis to logarithmic scale
# plt.ylim(0, 100)
plt.savefig('./data/plots/nafld_accuracy_by_features_log.png')
plt.show()
