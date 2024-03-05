import csv
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Charger les données depuis le fichier CSV
file_path = "annot_mouseStStAll.csv"

umap_1, umap_2, annot = [], [], []

with open(file_path, 'r') as csvfile:
    csvreader = csv.DictReader(csvfile)
    for row in csvreader:
        umap_1.append(float(row['UMAP_1']))
        umap_2.append(float(row['UMAP_2']))
        annot.append(row['annot'])

# Créer un dictionnaire pour mapper chaque valeur unique de "annot" à une couleur
unique_annot_values = list(set(annot))
color_mapping = {value: idx for idx, value in enumerate(unique_annot_values)}

# Convertir les valeurs de "annot" en couleurs avec une colormap différente ('tab20')
colors = [plt.cm.tab20(color_mapping[value]) for value in annot]

# Créer le graphique en nuage de points avec des couleurs différentes pour chaque valeur de "annot"
fig, ax = plt.subplots()
scatter = ax.scatter(umap_1, umap_2, c=colors)
ax.set_title('Human Cells')
ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')

# Ajouter une légende personnalisée
legend_labels = [(value, color_mapping[value]) for value in unique_annot_values]
legend_handles = [Line2D([0], [0], marker='o', color='w', label=f"{label}", 
                        markerfacecolor=plt.cm.tab20(color_mapping[label])) for label, _ in legend_labels]

# Positionner la légende à droite du graphique
ax.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left', title='Cells')

# Afficher le graphique
plt.show()
