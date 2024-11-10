import numpy as np

from source_code.spectral_equal_size_clustering import EqualSizeClustering

class WeightedEqualSizeClustering:
    """
    Postprocessor to be applied after clustering. Designed in a model agnostic manner, but refer disclosures.

    Disclosures:
    1. Developed primarily for my needs in KMeans clustering.
    2. Spectral clustering was also taken into account partially. 

    Notes:
    1. equity_fr can not be strictly enforced because of the weights (it's discrete jumps).

    Tasks:
    1. Attempt hacking to force it to diverge. Have max_iter as a safeguard.
    """

    def __init__(self, equity_fraction=0.4, max_expend_iter=50, max_steal_iter=50, batch_size=10):
        self.equity_fr = equity_fraction
        self.max_expend_iter = max_expend_iter
        self.max_steal_iter = max_steal_iter
        self.batch_size = batch_size

    def _iterate_equalization(self, dmatrix, weights, clustering, elements_per_cluster, larger_clusters, smaller_clusters, validate_and_switch):
        lnx = {c: list(clustering[clustering.label == c].index) for c in larger_clusters}
        snx = {c: list(clustering[clustering.label == c].index) for c in smaller_clusters}
        closest_distance = self._get_points_to_switch(dmatrix, weights, lnx, smaller_clusters, snx)
        
        batch_size = self.batch_size
        for point in list(closest_distance.index):
            if batch_size <= 0:
                break
            new_label = closest_distance.loc[point, "neighbor_c"]  # cluster that might receive the point
            current_label = clustering.loc[point, "label"]
            if not validate_and_switch(dmatrix, weights, clustering, current_label, new_label, point):
                continue
            batch_size -= weights[point]    

    def _iterate_expend_equalization(self, dmatrix, weights, clustering, elements_per_cluster, all_clusters, range_points):
        pass
    def cluster_equalization(self, dmatrix, weights, initial_labels):
        npoints = weights.sum()
        elements_per_cluster = EqualSizeClustering._optimal_cluster_sizes(self.nclusters, npoints)
        min_range = np.array(elements_per_cluster).min() * self.equity_fr
        max_range = np.array(elements_per_cluster).max() * self.equity_fr
        self.range_points = (min_range, max_range)

        all_clusters = list(np.arange(0, self.nclusters))
        clustering = self.first_clustering.copy()

        for _ in range(self.max_expend_iter):
            self._iterate_expend_equalization(dmatrix, weights, clustering, elements_per_cluster, all_clusters, self.range_points)

        for _ in range(self.max_steal_iter):
            self._iterate_expend_equalization(dmatrix, weights, clustering, elements_per_cluster, all_clusters, self.range_points)

        self.final_clustering = clustering
        self.final_cluster_dispersion = self._cluster_dispersion(dmatrix, weights, self.final_clustering)
        self.total_cluster_dispersion = self.final_cluster_dispersion["cdispersion"].sum(axis=0)


    def fit(self, dmatrix, weights, initial_labels):
        # self.cluster_initialization(dmatrix, initial_labels)
        self.cluster_equalization(dmatrix, weights, initial_labels)