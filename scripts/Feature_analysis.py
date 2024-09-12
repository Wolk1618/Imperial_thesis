import json
import matplotlib.pyplot as plt


# Load the features_importances JSON file
with open('data/wdsd_shap_values.json') as f:
    data = json.load(f)

features_list = []
# Print the head of the data
for data_item in data :
    features_list.append({data_item: data[data_item]})

# Sort the list of features by importance
sorted_features_list = sorted(features_list, key=lambda x: list(x.values())[0], reverse=True)

# # Print the 10 most important features
# for i in range(10):
#     print(list(sorted_features_list[i].keys())[0])

# Calculate the sum of features importance
sum_importance = sum(list(x.values())[0] for x in sorted_features_list)

# list_key_values = []

# # Calculate the sum of contribution of n most important features
# for n in range(10, 101, 10):
#     sum_top_n = sum(list(x.values())[0] for x in sorted_features_list[:n])
#     print("Sum of contribution of " + str(n) + " most important features:", sum_top_n)
#     list_key_values.append(list(sorted_features_list[n-1].values())[0])

# print("List of key importances : ", list_key_values)

# Group the features by 100
grouped_features = [sorted_features_list[i:i+100] for i in range(0, 401, 100)]
grouped_features.append(sorted_features_list[500:1000])
grouped_features.append(sorted_features_list[1000:])

# Calculate the sum of importance for each group
group_sums = [sum(list(x.values())[0] for x in group) for group in grouped_features]

# Create labels for the pie chart
labels = ['Group ' + str(i+1) for i in range(len(grouped_features) - 2)]
labels.append('Group 6-9')
labels.append('Group 10+')

# Plot the pie chart
plt.pie(group_sums, labels=labels, autopct='%1.1f%%')
plt.title('Features Importance by Group of 100')
plt.savefig('./data/plots/shap_features_importance_pie_chart.png')
plt.show()