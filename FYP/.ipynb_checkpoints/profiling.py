#import libraries
import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist

data = pd.read_csv('new_cleaned_dataset.csv')
data.head()

# Step 1 and 2 - Choose the number of clusters (k) and select random centroid for each cluster

#number of clusters
K=3

# Select random as centroids
X = data[["Spent","Income","Kidhome", "Teenhome", "Children", "Family_Size", "Is_Parent","Age","Living_With"]]
Centroids = (X.sample(n=K))
plt.scatter(X["Income"],X["Spent"],c='black')
plt.scatter(Centroids["Income"],Centroids["Spent"],c='red')
plt.xlabel('Income (Annual)')
plt.ylabel('Spent')
plt.show()


# Step 3 - Assign all the points to the closest cluster centroid

diff = 1
j = 0

while diff != 0:
    XD = X.copy()  # Create a copy of X for temporary values
    i = 1
    for index1, row_c in Centroids.iterrows():
        ED = []
        for index2, row_d in XD.iterrows():
            d1 = (row_c["Income"] - row_d["Income"])**2
            d2 = (row_c["Spent"] - row_d["Spent"])**2
            d = np.sqrt(d1 + d2)
            ED.append(d)
        X[i] = ED
        i = i + 1

    C = []
    for index, row in X.iterrows():
        min_dist = row[1]
        pos = 1
        for i in range(K):
            if row[i+1] < min_dist:
                min_dist = row[i+1]
                pos = i + 1
        C.append(pos)
    X["Cluster"] = C
    Centroids_new = X.groupby(["Cluster"]).mean()[["Spent", "Income"]]
    if j == 0:
        diff = 1
        j = j + 1
    else:
        diff = (Centroids_new['Spent'] - Centroids['Spent']).sum() + (Centroids_new['Income'] - Centroids['Income']).sum()
        print(diff.sum())
    Centroids = X.groupby(["Cluster"]).mean()[["Spent", "Income"]]

#Silhouette  score measure

def get_silhouette_score(X, Centroids):
    cluster_labels = []
    for index, row in X.iterrows():
        min_dist = float('inf')
        for centroid_index, centroid_row in Centroids.iterrows():
            distance = cdist([[row["Income"], row["Spent"]]], [[centroid_row["Income"], centroid_row["Spent"]]], 'euclidean')[0][0]
            if distance < min_dist:
                min_dist = distance
                best_centroid = centroid_index
        cluster_labels.append(best_centroid)
    return silhouette_score(X[["Income", "Spent"]], cluster_labels)

silhouette_score = get_silhouette_score(X, Centroids)
print(f"Silhouette Score: {silhouette_score}")

color=['blue','yellow','green']

# Plotting clusters with labels after assigning data points to their closest centroid
for k in range(K):
    cluster_data = X[X["Cluster"] == k+1]
    plt.scatter(cluster_data["Income"], cluster_data["Spent"], c=color[k], label=f'Cluster {k+1}')

plt.scatter(Centroids["Income"], Centroids["Spent"], c='red', marker='X', label='Centroids')
plt.xlabel('Income (Annual)')
plt.ylabel('Spent')
plt.legend()
plt.show()

#profiling
Personal = ["Family_Size", "Is_Parent","Children", "Teenhome","Kidhome","Living_With","Age"]

for i in Personal:
    plt.figure()
    sns.jointplot(x=X[i], y=X["Spent"], hue=X["Cluster"],kind='kde',palette=color)
    plt.show()