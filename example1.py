"""
This script shows how to run the spectral equal size clustering and regular spectral clustering side by side.
From a set of hyperparameters, you get clusters with size roughly equal to N/ncluster
"""
import pandas as pd
import numpy as np
import logging
from source_code.spectral_equal_size_clustering import SpectralEqualSizeClustering
from source_code.visualisation import visualise_clusters
from sklearn.cluster import SpectralClustering, KMeans
from source_code.weighted_equal_size_clustering import WeightedEqualSizeClustering

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

# read the file with coordinates. This file is used only for visualization purposes
coords = pd.read_csv("datasets/restaurants_in_amsterdam.csv")

# # read the file of the symmetric distance matrix associated to the coords data frame
# dist_tr = np.load("datasets/symmetric_dist_tr.npy")

# # Spectral Equal Size Clustering
# equal_size_clustering = SpectralEqualSizeClustering(nclusters=6,
#                                                     nneighbors=int(dist_tr.shape[0] * 0.1),
#                                                     equity_fraction=1,
#                                                     seed=1234)
# equal_size_labels = equal_size_clustering.fit(dist_tr)
# coords["equal_size_cluster"] = equal_size_labels
# logging.info(f"Points per equal size cluster: \n {coords.equal_size_cluster.value_counts()}")

# # Regular Spectral Clustering
# spectral_clustering = SpectralClustering(n_clusters=6,
#                                                 assign_labels="discretize",
#                                                 n_neighbors=int(dist_tr.shape[0] * 0.1),
#                                                 affinity="precomputed_nearest_neighbors",
#                                                 random_state=1234)
# regular_labels = spectral_clustering.fit_predict(dist_tr)
# coords["regular_cluster"] = regular_labels
# logging.info(f"Points per regular cluster: \n {coords.regular_cluster.value_counts()}")

print(coords.head())

coords = coords[["latitude", "longitude"]]

initial_clustering = KMeans(n_clusters=6, random_state=101)
initial_clustering.fit(coords[["latitude", "longitude"]])
initial_labels = initial_clustering.labels_
coords["kmeans_cluster"] = initial_labels
logging.info(f"Points per kmeans cluster: \n {coords.kmeans_cluster.value_counts()}")

# Weighted Equal Size Clustering
weighted_equal_size_clustering = WeightedEqualSizeClustering(nclusters=6,
                                                             equity_fraction=1,
                                                             max_expend_iter= 10,
                                                             max_steal_iter= 10
                                                             )
weighted_equal_size_labels = weighted_equal_size_clustering.fit(coords, np.ones(initial_labels.shape[0]), initial_labels)
coords["weighted_equal_size_cluster"] = weighted_equal_size_labels
logging.info(f"Points per weighted equal size cluster: \n {coords.weighted_equal_size_cluster.value_counts()}")

# # Equal Size Clustering Visualization
# equal_size_clusters_figure = visualise_clusters(coords,
#                                                 longitude_colname="longitude",
#                                                 latitude_colname="latitude",
#                                                 label_col="equal_size_cluster",
#                                                 zoom=11)
# equal_size_clusters_figure.show()

# # Regular Spectral Clustering Visualization
# regular_clusters_figure = visualise_clusters(coords,
#                                              longitude_colname="longitude",
#                                              latitude_colname="latitude",
#                                              label_col="regular_cluster",
#                                              zoom=11)
# regular_clusters_figure.show()

# KMeans Clustering Visualization
kmeasn_clusters_figure = visualise_clusters(coords,
                                                         longitude_colname="longitude",
                                                         latitude_colname="latitude",
                                                         label_col="kmeans_cluster",
                                                         zoom=11)
kmeasn_clusters_figure.show()


# Weighted Equal Size Clustering Visualization
weighted_equal_size_clusters_figure = visualise_clusters(coords,
                                                         longitude_colname="longitude",
                                                         latitude_colname="latitude",
                                                         label_col="weighted_equal_size_cluster",
                                                         zoom=11)
weighted_equal_size_clusters_figure.show()

print("Centroids: ")
print(weighted_equal_size_clustering.final_centroids)
