import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist

data = pd.read_csv('new_cleaned_dataset.csv')
data.head()

#number of clusters
K=5

# Select random as centroids

X = data[["Income","Spent","Age","Kidhome"]]
Centroids = (X.sample(n=K))

#Income Spent
diff = 1
j=0

while(diff!=0):
    XD=X
    i=1
    for index1,row_c in Centroids.iterrows():
        ED=[]
        for index2,row_d in XD.iterrows():
            d1=(row_c["Income"]-row_d["Income"])**2
            d2=(row_c["Spent"]-row_d["Spent"])**2
            d3=(row_c["Age"]-row_d["Age"])**2
            d4=(row_c["Kidhome"]-row_d["Kidhome"])**2
            d=np.sqrt(d1+d2+d3+d4)
            ED.append(d)
        X[i]=ED
        i=i+1

    C=[]
    for index,row in X.iterrows():
        min_dist=row[1]
        pos=1
        for i in range(K):
            if row[i+1] < min_dist:
                min_dist = row[i+1]
                pos=i+1
        C.append(pos)
    X["Cluster"]=C
    Centroids_new = X.groupby(["Cluster"]).mean()[["Spent","Income","Age","Kidhome"]]
    if j == 0:
        diff=1
        j=j+1
    else:
        diff = (Centroids_new['Spent'] - Centroids['Spent']).sum() + (Centroids_new['Income'] - Centroids['Income']).sum()+ (Centroids_new['Age'] - Centroids['Age']).sum()+ (Centroids_new['Kidhome'] - Centroids['Kidhome']).sum()
        print(diff.sum())
    Centroids = X.groupby(["Cluster"]).mean()[["Spent","Income","Age","Kidhome"]]

#Silhouette score measure

def get_silhouette_score(X, Centroids):
    cluster_labels = []
    for index, row in X.iterrows():
        min_dist = float('inf')
        for centroid_index, centroid_row in Centroids.iterrows():
            distance = cdist([[row["Income"], row["Spent"],row["Age"],row["Kidhome"]]], [[centroid_row["Income"], centroid_row["Spent"],centroid_row["Age"],centroid_row["Kidhome"]]], 'minkowski', p=2)[0][0]
            if distance < min_dist:
                min_dist = distance
                best_centroid = centroid_index
        cluster_labels.append(best_centroid)
    return silhouette_score(X[["Income", "Spent","Age","Kidhome"]], cluster_labels)


silhouette_score = get_silhouette_score(X, Centroids)
print(f"Silhouette Score: {silhouette_score}")