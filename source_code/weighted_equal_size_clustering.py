import numpy as np
import pandas as pd

from source_code.spectral_equal_size_clustering import EqualSizeClustering

class WeightedEqualSizeClustering:
    """
    Postprocessor to be applied after clustering. Designed in a model agnostic manner, but refer disclosures.

    Architecture:
    1. This class can inherit from EqualSizeClustering, as it's using its methods anyway. 
        Solution: Better work on this class alone with full dedication, as the original repo is not maintained

    Disclosures:
    1. Developed primarily for my needs in KMeans clustering.
    2. Spectral clustering was also taken into account partially. 

    Notes:
    1. equity_fr can not be strictly enforced because of the weights (it's discrete jumps).

    Tasks:
    1. Attempt hacking to force it to diverge. Have max_iter as a safeguard.

    Edge Cases:
    1. There could be a case where the best action is for a large cluster A to donate to an okay cluster B, even though this donation makes B become a large cluster, if B can later donate another of its point to another cluster

    Future Improvements:
    1. If we have access to the points itself, we can cache the centroids instead of having to calculate from the dmatrix each time.
    """

    def __init__(self, nclusters, equity_fraction=0.4, max_expend_iter=50, max_steal_iter=50, batch_size=10, point_to_cluster_calculator=None):
        self.nclusters = nclusters
        self.equity_fr = equity_fraction
        self.max_expend_iter = max_expend_iter
        self.max_steal_iter = max_steal_iter
        self.batch_size = batch_size

        self.point_to_cluster_calculator = point_to_cluster_calculator
        if self.point_to_cluster_calculator is None:
            self.point_to_cluster_calculator = self._weighted_point_to_cluster

    @staticmethod
    def _current_elements_per_cluster(clustering, weights):
        clusters = clustering.label.unique()
        return [weights[clustering[clustering.label == c].index].sum() for c in clusters]

    @staticmethod
    def _get_clusters_outside_range(clustering, weights, minr, maxr):
        clusters = clustering.label.unique() # assuming unique() and value_counts() use the same sequence

        csizes = pd.DataFrame({
            "cluster": clusters,
            "npoints": WeightedEqualSizeClustering._current_elements_per_cluster(clustering, weights)
        })

        large_c = list(csizes[csizes.npoints > maxr]["cluster"].values)
        small_c = list(csizes[csizes.npoints < minr]["cluster"].values)

        return large_c, small_c
    
    @staticmethod
    def _weighted_point_to_cluster(dmatrix, weights, cluster, point):
        return (dmatrix[cluster, point] * weights[cluster]).mean()

    def _get_points_to_switch(self, weights, dmatrix, cl_elements, clusters_to_modify, idxc):
        centroids = self.get_centroids(dmatrix, weights, idxc)
        neighbor_cluster = []
        distances = []
        for point in cl_elements:
            # dist = {c: dmatrix[idxc[c], point].mean() for c in clusters_to_modify}  # Instead of min. Worth future inv.
            dist = {c: self.point_to_cluster_calculator(dmatrix, weights, idxc[c], point) for c in clusters_to_modify}
            new_label = min(dist, key=dist.get)  # closest cluster
            neighbor_cluster.append(new_label)
            distances.append(dist[new_label])

        cdistances = pd.DataFrame({"points": cl_elements, "neighbor_c": neighbor_cluster, "distance": distances})
        cdistances = cdistances.sort_values(by="distance", ascending=True).set_index("points")
        return cdistances

    def _iterate_equalization(self, dmatrix, weights, clustering, current_elements_per_cluster, larger_clusters, smaller_clusters, thresholds):
        def validate_and_switch(weights, clustering, current_elements_per_cluster, current_label, new_label, point):
            weight = weights[point]
            if current_elements_per_cluster[current_label] - weight < thresholds[current_label]:
                return False
            if current_elements_per_cluster[new_label] + weight > thresholds[new_label]:
                return False
            clustering.loc[point, "label"] = new_label
            current_elements_per_cluster[current_label] -= weight
            current_elements_per_cluster[new_label] += weight
            return True

        larger_cluster_points = clustering[clustering.label in larger_clusters].index
        snx = {c: list(clustering[clustering.label == c].index) for c in smaller_clusters}
        closest_distance = self._get_points_to_switch(dmatrix, weights, larger_cluster_points, smaller_clusters, snx)

        batch_size = self.batch_size
        for point in list(closest_distance.index):
            if batch_size <= 0:
                break
            new_label = closest_distance.loc[point, "neighbor_c"]  # cluster that might receive the point
            current_label = clustering.loc[point, "label"]
            if not validate_and_switch(weights, clustering, current_elements_per_cluster, current_label, new_label, point):
                continue
            batch_size -= weights[point]    

    def cluster_initialization(self, dist_matrix, initial_labels):
        self.first_clustering = pd.DataFrame(initial_labels, columns=["label"])
        # self.first_cluster_dispersion = self._cluster_dispersion(dist_matrix, self.first_clustering)
        # self.first_total_cluster_dispersion = self.first_cluster_dispersion["cdispersion"].sum()


    def cluster_equalization(self, dmatrix, weights):
        npoints = weights.sum()
        optimal_elements_per_cluster = EqualSizeClustering._optimal_cluster_sizes(self.nclusters, npoints)
        min_range = np.array(optimal_elements_per_cluster).min() * self.equity_fr
        max_range = np.array(optimal_elements_per_cluster).max() * self.equity_fr
        self.range_points = (min_range, max_range)

        all_clusters = list(np.arange(0, self.nclusters))
        clustering = self.first_clustering.copy()

        large_clusters, small_clusters = self._get_clusters_outside_range(clustering, weights, min_range, max_range)
 
        # Note "large" and "larger" and "small" and "smaller" 

        current_elements_per_cluster = self._current_elements_per_cluster(clustering, weights)

        common_args = (dmatrix, weights, clustering, current_elements_per_cluster)

        for _ in range(self.max_expend_iter):
            larger_clusters = large_clusters
            smaller_clusters = list(set(all_clusters) - set(large_clusters))  # clusters that receive points
            self._iterate_equalization(
                *common_args, 
                larger_clusters, 
                smaller_clusters, 
                max_range
            )

        for _ in range(self.max_steal_iter):
            larger_clusters = list(set(all_clusters) - set(small_clusters))
            smaller_clusters = small_clusters
            self._iterate_equalization(
                *common_args, 
                larger_clusters, 
                smaller_clusters, 
                min_range
            )

        self.final_clustering = clustering
        # self.final_cluster_dispersion = self._cluster_dispersion(dmatrix, weights, self.final_clustering)
        # self.total_cluster_dispersion = self.final_cluster_dispersion["cdispersion"].sum(axis=0)


    def fit(self, dmatrix, weights, initial_labels):
        if self.nclusters == np.shape(dmatrix)[0]:
            raise Exception("Number of clusters equal to number of events.")

        if self.nclusters <= 1:
            raise ValueError("Incorrect number of clusters. It should be higher or equal than 2.")

        self.cluster_initialization(dmatrix, initial_labels)
        self.cluster_equalization(dmatrix, weights)

        return list(self.final_clustering.label.values)