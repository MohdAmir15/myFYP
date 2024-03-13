# Import libraries
import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist

# Read the dataset
data = pd.read_csv('new_cleaned_dataset.csv')
data.head()

# Choose the number of clusters (k) and select random centroid for each cluster
K = 5

# Select random centroids
X = data[["Income", "Spent", "Age","Kidhome","Teenhome","Family_Size"]]
Centroids = (X.sample(n=K))

# Define Manhattan distance function
def manhattan_distance(point1, point2):
    return np.abs(point1 - point2).sum()

# Iterate until convergence
diff = 1
j = 0

while diff != 0:
    XD = X.copy()
    i = 1

    # Calculate Manhattan distance
    for index1, row_c in Centroids.iterrows():
        ED = []
        for index2, row_d in XD.iterrows():
            d = manhattan_distance(row_c, row_d)
            ED.append(d)
        X[i] = ED
        i += 1

    # Assign points to the closest cluster centroid
    C = []
    for index, row in X.iterrows():
        min_dist = row[1]
        pos = 1
        for i in range(K):
            if row[i + 1] < min_dist:
                min_dist = row[i + 1]
                pos = i + 1
        C.append(pos)
    X["Cluster"] = C

    # Recompute centroids of newly formed clusters
    Centroids_new = X.groupby(["Cluster"]).mean()[["Spent", "Income", "Age", "Kidhome","Teenhome","Family_Size"]]

    if j == 0:
        diff = 1
        j += 1
    else:
        # Calculate the difference in centroids
        diff = (Centroids_new['Spent'] - Centroids['Spent']).sum() + \
               (Centroids_new['Income'] - Centroids['Income']).sum() + \
               (Centroids_new['Age'] - Centroids['Age']).sum() + \
               (Centroids_new['Kidhome'] - Centroids['Kidhome']).sum() + \
               (Centroids_new['Teenhome'] - Centroids['Teenhome']).sum() + \
               (Centroids_new['Family_Size'] - Centroids['Family_Size']).sum()
        print(diff.sum())

    Centroids = X.groupby(["Cluster"]).mean()[["Spent", "Income", "Age", "Kidhome", "Teenhome", "Family_Size"]]

# Silhouette score measure
def get_silhouette_score(X, Centroids):
    cluster_labels = []
    for index, row in X.iterrows():
        min_dist = float('inf')
        for centroid_index, centroid_row in Centroids.iterrows():
            distance = cdist([[row["Income"], row["Spent"], row["Age"], row["Kidhome"], row["Teenhome"], row["Family_Size"]]],
                              [[centroid_row["Income"], centroid_row["Spent"], centroid_row["Age"],centroid_row["Kidhome"],centroid_row["Teenhome"],centroid_row["Family_Size"]]],
                              'cityblock')[0][0]  # Use 'cityblock' for Manhattan distance
            if distance < min_dist:
                min_dist = distance
                best_centroid = centroid_index
        cluster_labels.append(best_centroid)
    return silhouette_score(X[["Income", "Spent", "Age", "Kidhome", "Teenhome", "Family_Size"]], cluster_labels)

silhouette_score = get_silhouette_score(X, Centroids)
print(f"Silhouette Score: {silhouette_score}")
