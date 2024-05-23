from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from pyclust import KMedoids
from pyclustering.cluster.kmedoids import kmedoids
from tqdm import tqdm

from tinyfaces.clustering.k_medoids import kMedoids
from tinyfaces.metrics import jaccard_index, rect_dist


def centralize_bbox(bboxes):
    """
    Convert the bounding boxes from (x, y, w, h) to (-w/2, -h/2, w/2, h/2).
    We perform clustering based on aspect ratio only.
    """
    print("Centralize and vectorize")
    hs = bboxes[:, 3] - bboxes[:, 1] + 1
    ws = bboxes[:, 2] - bboxes[:, 0] + 1
    rects = np.vstack(
        [-(ws - 1) / 2, -(hs - 1) / 2, (ws - 1) / 2, (hs - 1) / 2]).T

    return rects


def compute_distances(bboxes):
    print("Computing distances")
    distances = np.zeros((len(bboxes), len(bboxes)))
    for i in tqdm(range(len(bboxes)), total=len(bboxes)):
        for j in range(len(bboxes)):
            distances[i, j] = 1 - jaccard_index(bboxes[i, :], bboxes[j, :],
                                                (i, j))

    return distances


def compute_kmedoids(bboxes,
                     cls,
                     option='pyclustering',
                     indices=15,
                     max_clusters=35,
                     max_limit=5000):
    print("Performing clustering using", option)
    clustering = [{} for _ in range(indices)]

    bboxes = centralize_bbox(bboxes)

    # subsample the number of bounding boxes so that it can fit in memory and is faster
    if bboxes.shape[0] > max_limit:
        sub_ind = np.random.choice(np.arange(bboxes.shape[0]),
                                   size=max_limit,
                                   replace=False)
        bboxes = bboxes[sub_ind]

    distances_cache = Path('distances_{0}.jbl'.format(cls))
    if distances_cache.exists():
        print("Loading distances")
        dist = joblib.load(distances_cache)
    else:
        dist = compute_distances(bboxes)
        joblib.dump(dist, distances_cache, compress=5)

    if option == 'pyclustering':
        for k in range(indices, max_clusters + 1):
            print(k, "clusters")

            initial_medoids = np.random.choice(bboxes.shape[0],
                                               size=k,
                                               replace=False)

            kmedoids_instance = kmedoids(dist,
                                         initial_medoids,
                                         ccore=True,
                                         data_type='distance_matrix')

            print("Running KMedoids")
            t1 = datetime.now()
            kmedoids_instance.process()
            dt = datetime.now() - t1
            print("Total time taken for clustering {k} medoids: {0}min:{1}s".
                  format(dt.seconds // 60, dt.seconds % 60, k=k))

            medoids_idx = kmedoids_instance.get_medoids()
            medoids = bboxes[medoids_idx]

            clustering.append({
                'n_clusters': k,
                'medoids': medoids,
                'class': cls
            })

    elif option == 'pyclust':

        for k in range(indices, max_clusters + 1):
            print(k, "clusters")
            kmd = KMedoids(n_clusters=k,
                           distance=rect_dist,
                           n_trials=1,
                           max_iter=2)
            t1 = datetime.now()
            kmd.fit(bboxes)
            dt = datetime.now() - t1
            print("Total time taken for clustering {k} medoids: {0}min:{1}s".
                  format(dt.seconds // 60, dt.seconds % 60, k=k))

            medoids = kmd.centers_

            clustering.append({
                'n_clusters': k,
                'medoids': medoids,
                'class': cls
            })

    elif option == 'local':

        for k in range(indices, max_clusters + 1):
            print(k, "clusters")
            curr_medoids, cluster_idxs = kMedoids(dist, k=k)
            medoids = []
            for m in curr_medoids:
                medoids.append(bboxes[m, :])
            clustering.append({
                'n_clusters': k,
                'medoids': medoids,
                'class': cls
            })

    return clustering
