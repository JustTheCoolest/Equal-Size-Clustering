import numpy as np
import pandas as pd
import math

from source_code.spectral_equal_size_clustering import SpectralEqualSizeClustering

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
    2. coords implementation seems to be running slower than dmatrix. Analyse time complexities.

    Edge Cases:
    1. There could be a case where the best action is for a large cluster A to donate to an okay cluster B, even though this donation makes B become a large cluster, if B can later donate another of its point to another cluster

    Future Improvements:
    1. If we have access to the points itself, we can cache the centroids instead of having to calculate from the dmatrix each time.
    """

    def __init__(self, nclusters, equity_fraction=0.4, max_expend_iter=50, max_steal_iter=50, batch_size=10, point_to_cluster_calculator=None, centroid_calculator=None):
        self.nclusters = nclusters
        self.equity_fr = equity_fraction
        self.max_expend_iter = max_expend_iter
        self.max_steal_iter = max_steal_iter
        self.batch_size = batch_size

        self.point_to_cluster_calculator = point_to_cluster_calculator
        if self.point_to_cluster_calculator is None:
            self.point_to_cluster_calculator = self._weighted_point_to_cluster
        
        self.centroid_calculator = centroid_calculator
        if self.centroid_calculator is None:
            self.centroid_calculator = self._calculate_centroid

    @staticmethod
    def _optimal_cluster_sizes(nclusters, npoints):
        min_points, max_points = math.floor(npoints / float(nclusters)), math.floor(npoints / float(nclusters)) + 1
        number_clusters_with_max_points = npoints % nclusters
        number_clusters_with_min_points = nclusters - number_clusters_with_max_points

        # print(npoints, nclusters, number_clusters_with_max_points)
        assert number_clusters_with_max_points == int(number_clusters_with_max_points)
        number_clusters_with_max_points = int(number_clusters_with_max_points)
        assert number_clusters_with_min_points == int(number_clusters_with_min_points)
        number_clusters_with_min_points = int(number_clusters_with_min_points)

        list1 = list(max_points * np.ones(number_clusters_with_max_points).astype(int))
        list2 = list(min_points * np.ones(number_clusters_with_min_points).astype(int))
        return list1 + list2

    @staticmethod
    def _current_elements_per_cluster(clustering, weights):
        clusters = clustering.label.unique()
        return {c: weights[clustering[clustering.label == c].index].sum() for c in clusters}
    
    @staticmethod
    def _get_clusters_outside_range(clustering, weights, minr, maxr):
        clusters = clustering.label.unique() # assuming unique() and value_counts() use the same sequence

        csizes = pd.DataFrame({
            "cluster": clusters,
            "npoints": [weights[clustering[clustering.label == c].index].sum() for c in clusters]
        })

        large_c = list(csizes[csizes.npoints > maxr]["cluster"].values)
        small_c = list(csizes[csizes.npoints < minr]["cluster"].values)

        return large_c, small_c
    
    @staticmethod
    def _calculate_centroid(X, weights, idxc):
        selected_X = X[idxc]
        selected_weights = weights[idxc]
        selected_weights_2d = selected_weights[:, np.newaxis]
        weighted_X = selected_X * selected_weights_2d
        weighted_sum = weighted_X.sum(axis=0)
        total_weight = selected_weights.sum()
        centroid = weighted_sum / total_weight        
        return centroid
    
    def _calculate_centroids(self, X, weights, clusters, idxc):
        return {c: self.centroid_calculator(X, weights, idxc[c]) for c in clusters}
    
    @staticmethod
    def _weighted_point_to_cluster(X, weights, cluster, point, centroid):
        return np.sqrt(((X[point] - centroid) ** 2).sum())
    
    def _get_weighted_distances(self, X, weights, idxc, point, centroids, clusters):
        dist = {}
        for c in clusters:
            idxc_c = idxc[c]
            centroid_c = centroids[c]
            distance = self.point_to_cluster_calculator(X, weights, idxc_c, point, centroid_c)
            dist[c] = distance
        return dist

    def _get_points_to_switch(self, X, weights, cl_elements, clusters_to_modify, idxc):
        centroids = self._calculate_centroids(X, weights, clusters_to_modify, idxc)

        neighbor_cluster = []
        distances = []
        for point in cl_elements:
            # Weight of point doesn't affect its chance of being moved
            dist = self._get_weighted_distances(X, weights, idxc, point, centroids, clusters_to_modify)
            new_label = min(dist, key=dist.get)  # closest cluster
            neighbor_cluster.append(new_label)
            distances.append(dist[new_label])

        cdistances = pd.DataFrame({"points": cl_elements, "neighbor_c": neighbor_cluster, "distance": distances})
        cdistances = cdistances.sort_values(by="distance", ascending=True).set_index("points")
        return cdistances

    def _iterate_equalization(self, X, weights, clustering, current_elements_per_cluster, larger_clusters, smaller_clusters, threshold):
        def validate_and_switch(weights, clustering, current_elements_per_cluster, current_label, new_label, point):
            weight = weights[point]
            # note: recheck edge case logic
            if current_elements_per_cluster[current_label] - weight < threshold:
                return False
            if current_elements_per_cluster[new_label] + weight > threshold:
                return False
            clustering.loc[point, "label"] = new_label
            current_elements_per_cluster[current_label] -= weight
            current_elements_per_cluster[new_label] += weight
            return True

        larger_cluster_points = clustering[clustering.label.isin(larger_clusters)].index
        snx = {c: list(clustering[clustering.label == c].index) for c in smaller_clusters}
        closest_distance = self._get_points_to_switch(X, weights, larger_cluster_points, smaller_clusters, snx)

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
        self.clusters_index = self.first_clustering.label.unique()
        # self.first_cluster_dispersion = self._cluster_dispersion(dist_matrix, self.first_clustering)
        # self.first_total_cluster_dispersion = self.first_cluster_dispersion["cdispersion"].sum()


    def cluster_equalization(self, X, weights):
        npoints = weights.sum()
        optimal_elements_per_cluster = self._optimal_cluster_sizes(self.nclusters, npoints)
        min_range = np.array(optimal_elements_per_cluster).min() * self.equity_fr
        max_range = np.array(optimal_elements_per_cluster).max() * self.equity_fr
        self.range_points = (min_range, max_range)

        all_clusters = list(np.arange(0, self.nclusters))
        clustering = self.first_clustering.copy()

        large_clusters, small_clusters = self._get_clusters_outside_range(clustering, weights, min_range, max_range)
 
        # Note "large" and "larger" and "small" and "smaller" 

        current_elements_per_cluster = self._current_elements_per_cluster(clustering, weights)

        common_args = (X, weights, clustering, current_elements_per_cluster)

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
        # self.final_cluster_dispersion = self._cluster_dispersion(X, weights, self.final_clustering)
        # self.total_cluster_dispersion = self.final_cluster_dispersion["cdispersion"].sum(axis=0)


    def fit(self, X, weights, initial_labels):
        if type(X) == pd.DataFrame:
            X = X.to_numpy()
        # CRITICAL FLAG: Document behaviour, and/or make it a preprocessing step for all the inputs in a logically correct manner

        if self.nclusters == np.shape(X)[0]:
            raise Exception("Number of clusters equal to number of events.")

        if self.nclusters <= 1:
            raise ValueError("Incorrect number of clusters. It should be higher or equal than 2.")

        self.cluster_initialization(X, initial_labels)
        self.cluster_equalization(X, weights)

        self.final_centroids = self._calculate_centroids(X, weights, self.clusters_index, {c: list(self.final_clustering[self.final_clustering.label == c].index) for c in self.clusters_index}).values()

        return list(self.final_clustering.label.values)
