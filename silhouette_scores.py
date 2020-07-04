import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples
from sklearn.cluster import KMeans, AgglomerativeClustering


def plot_avg_silhouette_score_and_get_cluster_with_max_score(clustering_method, features, cluster_list):

    avg_scores = {}

    for i, k in enumerate(cluster_list):

        fig, (ax1) = plt.subplots(1, 1)
        fig.set_size_inches(18, 7)

        if(clustering_method == 'KMeans'):
            model = KMeans(n_clusters = k, random_state=0)

        if(clustering_method == 'AgglomerativeClustering'):
            model = AgglomerativeClustering(n_clusters = k)


        labels = model.fit_predict(features)
        
        # Get silhouette samples
        silhouette_vals = silhouette_samples(features, labels)

        # Silhouette plot
        y_ticks = []
        y_lower, y_upper = 0, 0
        for i, cluster in enumerate(np.unique(labels)):
            cluster_silhouette_vals = silhouette_vals[labels == cluster]
            cluster_silhouette_vals.sort()
            y_upper += len(cluster_silhouette_vals)
            ax1.barh(range(y_lower, y_upper), cluster_silhouette_vals, edgecolor='none', height=1)
            ax1.text(-0.03, (y_lower + y_upper) / 2, str(i + 1))
            y_lower += len(cluster_silhouette_vals)

        # Get the average silhouette score and plot it
        avg_score = np.mean(silhouette_vals)
        ax1.axvline(avg_score, linestyle='--', linewidth=2, color='green')
        ax1.set_yticks([])
        ax1.set_xlim([-0.1, 1])
        ax1.set_xlabel('Silhouette coefficient values')
        ax1.set_ylabel('Cluster labels')
        ax1.set_title('Silhouette plot for the various clusters', y=1.0)

        avg_scores[k] = avg_score

    cluster_with_max_score = max(avg_scores, key=avg_scores.get)
    return (cluster_with_max_score, avg_scores[cluster_with_max_score]) 
