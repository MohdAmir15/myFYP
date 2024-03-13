#import libraries
import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
import pickle

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

# Save the model
pickle.dump(Centroids, open('clusteringmodel', 'wb'))