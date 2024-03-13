# Dashboard coding
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist

# Load the k-means clustering model
model = pickle.load(open('clustering_model.pkl', 'rb'))

def scatter_plot(data, var_x, var_y=None):
    plt.figure()
    if var_y:
        plt.scatter(data[var_x], data[var_y])
        plt.ylabel(var_y)
        plt.title(f'Scatter Plot: {var_x} vs {var_y}')
    else:
        plt.scatter(data[var_x])
        plt.title(f'Scatter Plot: {var_x}')
    plt.xlabel(var_x)
    return plt.gcf()

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

def run_clustering(data, selected_attribute_x):
    # Load the customer data
    customer_data = pd.read_csv('new_cleaned_dataset.csv')

    # Copy the data to avoid modifying the original DataFrame
    X = customer_data.copy()

    # Number of clusters
    K = 3

    # Select random centroids
    Centroids = (X.sample(n=K))

    # Assign all the points to the closest cluster centroid
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

    # Create a DataFrame to store cluster information
    cluster_info = pd.DataFrame({
        'Cluster 1': [
            'Low-income group',
            'Max family member of 5 and at least 1',
            'Most are parents',
            'Max 3 children or none',
            'Mostly married',
            'Relatively younger'
        ],
        'Cluster 2': [
            'Moderate income group',
            'Max family member of 4 and at least 1',
            'Most are not parents',
            'Mostly have a teenager at home',
            'Mostly married',
            'Relatively younger'
        ],
        'Cluster 3': [
            'High income group',
            'Max family member of 4 and at least 1',
            'Most are parents',
            'Mostly have a teenager and a kid at home',
            'Mostly married',
            'Span all ages'
        ]
    }, index=['', '', '', '', '', ''])

    # Display the scatter plot with centroids and cluster labels
    plt.figure()
    color = ['blue', 'yellow', 'green']
    for k in range(K):
        cluster_data = X[X["Cluster"] == k+1]
        plt.scatter(cluster_data[selected_attribute_x], cluster_data["Spent"], c=color[k], label=f'Cluster {k+1}')

    plt.xlabel(selected_attribute_x)
    plt.ylabel('Spent')
    plt.legend()
    st.pyplot(plt.gcf())
    plt.close()

    st.markdown("### Kernel Density Estimation (KDE) Plot for Each Cluster")

    # Placeholder for KDE plots
    kde_plot_placeholders = [st.empty() for _ in range(K)]

    # KDE plot
    Personal = ["Family_Size", "Is_Parent", "Children", "Teenhome", "Kidhome", "Living_With", "Age", "Income"]
    # selected_attribute_x is used here as well
    for k in range(K):
        cluster_data = X[X["Cluster"] == k+1]
        plt.figure()
        sns.kdeplot(data=cluster_data, x=selected_attribute_x, y='Spent', hue='Cluster', palette=color, fill=True)
        plt.title(f'Cluster {k+1}')
        kde_plot_placeholders[k].pyplot(plt.gcf())

    # Display the cluster information table
    st.markdown("### Cluster Information Table")
    st.table(cluster_info)

    st.markdown("### Model Evaluation Metrics")

    # Calculate silhouette score using the silhouette_score function
    silhouette = get_silhouette_score(X, Centroids)
    st.info(f'Silhouette Score: {silhouette}')

def main():
    st.markdown("# Customer Segmentation Using K-Means Algorithm Dashboard")
    st.markdown('<style>description{color:blue;}</style>', unsafe_allow_html=True)
    st.sidebar.markdown("## Clustering Analysis Selection")
    step = st.sidebar.radio("", ["Features", "Clusters"])
    if step == "Features":
        run_features()
    elif step == "Clusters":
        # Load your customer data for clustering analysis
        data = pd.read_csv('new_cleaned_dataset.csv')
        st.markdown("### Scatter Plots (Clustered)")
        # Get the selected x-axis attribute for both plots
        selected_attribute_x = st.selectbox('Select x-axis attribute for plots', ["Family_Size", "Is_Parent", "Children", "Teenhome", "Kidhome", "Living_With", "Age", "Income"])
        run_clustering(data, selected_attribute_x)

def run_features():
    # Load your customer data for feature analysis
    data = pd.read_csv('new_cleaned_dataset.csv')
    st.markdown("### Customer Data")
    st.write(data)
    st.markdown("### Scatter Plots")
    # Scatter plot for selected features
    var_x_options = data.columns[1:]
    var_x = st.selectbox('Select x-axis attribute', var_x_options)
    var_y = st.selectbox('Select y-axis attribute', var_x_options)
    st.pyplot(scatter_plot(data, var_x, var_y))

if __name__ == "__main__":
    main()
