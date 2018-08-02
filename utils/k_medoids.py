import numpy as np
import warnings


def kMedoids(distances, k):
    """
    https://github.com/salspaugh/machine_learning/blob/master/clustering/kmedoids.py

    :param distances:
    :param k:
    :return:
    """
    n = distances.shape[0]
    medoid_idxs = np.random.choice(n, size=k, replace=False)
    old_medoids_idxs = np.zeros(k)

    while not np.all(medoid_idxs == old_medoids_idxs): # and n_iter_ < max_iter_
        # retain a copy of the old assignments
        old_medoids_idxs = np.copy(medoid_idxs)

        cluster_idxs = get_cluster_indices(distances, medoid_idxs)

        medoid_idxs = update_medoids(distances, cluster_idxs, medoid_idxs)

    return medoid_idxs, cluster_idxs


def get_cluster_indices(distances, medoid_idxs):
    cluster_idxs = np.argmin(distances[medoid_idxs, :], axis=0)
    return cluster_idxs


def update_medoids(distances, cluster_idxs, medoid_idxs):
    for cluster_idx in range(medoid_idxs.shape[0]):
        if sum(cluster_idxs == cluster_idx) == 0:
            warnings.warn("Cluster {} is empty!".format(cluster_idx))
            continue

        curr_cost = np.sum(distances[medoid_idxs[cluster_idx], cluster_idxs == cluster_idx])

        # Extract the distance matrix between the data points
        # inside the cluster_idx
        D_in = distances[cluster_idxs == cluster_idx, :]
        D_in = D_in[:, cluster_idxs == cluster_idx]

        # Calculate all costs there exists between all
        # the data points in the cluster_idx
        all_costs = np.sum(D_in, axis=1)

        # Find the index for the smallest cost in cluster_idx
        min_cost_idx = np.argmin(all_costs)

        # find the value of the minimum cost in cluster_idx
        min_cost = all_costs[min_cost_idx]

        # If the minimum cost is smaller than that
        # exhibited by the currently used medoid,
        # we switch to using the new medoid in cluster_idx
        if min_cost < curr_cost:
            # Find data points that belong to cluster_idx,
            # and assign the newly found medoid as the medoid
            # for cluster c
            medoid_idxs[cluster_idx] = np.where(cluster_idxs == cluster_idx)[0][min_cost_idx]

    return medoid_idxs
