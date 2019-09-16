
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from pyclust import KMedoids
from pyclustering.cluster.kmedoids import kmedoids
import joblib
from tqdm import tqdm

from .k_medoids import kMedoids
from .metrics import jaccard_index, rect_dist


def centralize_bbox(bboxes):
    """
    Convert the bounding boxes from (x, y, w, h) to (-w/2, -h/2, w/2, h/2).
    We perform clustering based on aspect ratio only.
    """
    print("Centralize and vectorize")
    hs = bboxes[:, 3] - bboxes[:, 1] + 1
    ws = bboxes[:, 2] - bboxes[:, 0] + 1
    rects = np.vstack([-(ws-1)/2, -(hs-1)/2, (ws-1)/2, (hs-1)/2]).T

    return rects


def compute_distances(bboxes):
    print("Computing distances")
    distances = np.zeros((len(bboxes), len(bboxes)))
    for i in tqdm(range(len(bboxes)), total=len(bboxes)):
        for j in range(len(bboxes)):
            distances[i, j] = 1 - jaccard_index(bboxes[i, :], bboxes[j, :], (i, j))

    return distances


def draw_bboxes(clusters):
    """
    Draw and save the clustered bounding boxes for inspection
    :param clusters:
    :return:
    """
    im = Image.new('RGB', [512, 512])
    d = ImageDraw.Draw(im)

    for bbox in clusters['medoids']:
        box = [(0, 0), (-bbox[0]+bbox[2], -bbox[1]+bbox[3])]
        color = tuple(np.random.choice(range(256), size=3))
        d.rectangle(box, outline=color)

    im.save("canonical_bbox_clusters_{0}.jpg".format(len(clusters['medoids'])))
    # im.show()


def compute_kmedoids(bboxes, cls, option='pyclustering', indices=15, max_clusters=35, max_limit=5000):
    print("Performing clustering using", option)
    clustering = [{} for _ in range(indices)]

    bboxes = centralize_bbox(bboxes)

    # subsample the number of bounding boxes so that it can fit in memory and is faster
    if bboxes.shape[0] > max_limit:
        sub_ind = np.random.choice(np.arange(bboxes.shape[0]), size=max_limit, replace=False)
        bboxes = bboxes[sub_ind]

    distances_cache = Path('distances_{0}.jbl'.format(cls))
    if distances_cache.exists():
        print("Loading distances")
        dist = joblib.load(distances_cache)
    else:
        dist = compute_distances(bboxes)
        joblib.dump(dist, distances_cache, compress=5)

    if option == 'pyclustering':
        for k in range(indices, max_clusters+1):
            print(k, "clusters")

            initial_medoids = np.random.choice(bboxes.shape[0], size=k, replace=False)

            kmedoids_instance = kmedoids(dist, initial_medoids, ccore=True, data_type='distance_matrix')

            print("Running KMedoids")
            t1 = datetime.now()
            kmedoids_instance.process()
            dt = datetime.now() - t1
            print("Total time taken for clustering {k} medoids: {0}min:{1}s"
                  .format(dt.seconds // 60, dt.seconds % 60, k=k))

            medoids_idx = kmedoids_instance.get_medoids()
            medoids = bboxes[medoids_idx]

            clustering.append({'n_clusters': k, 'medoids': medoids, 'class': cls})

    elif option == 'pyclust':

        for k in range(indices, max_clusters+1):
            print(k, "clusters")
            kmd = KMedoids(n_clusters=k, distance=rect_dist, n_trials=1, max_iter=2)
            t1 = datetime.now()
            kmd.fit(bboxes)
            dt = datetime.now() - t1
            print("Total time taken for clustering {k} medoids: {0}min:{1}s"
                  .format(dt.seconds//60, dt.seconds % 60, k=k))

            medoids = kmd.centers_

            clustering.append({'n_clusters': k, 'medoids': medoids, 'class': cls})

    elif option == 'local':

        for k in range(indices, max_clusters+1):
            print(k, "clusters")
            curr_medoids, cluster_idxs = kMedoids(dist, k=k)
            medoids = []
            for m in curr_medoids:
                medoids.append(bboxes[m, :])
            clustering.append({'n_clusters': k, 'medoids': medoids, 'class': cls})

    return clustering


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path')
    # 3 is the category ID for cars
    parser.add_argument('--cls', default=3, type=int, help="Indicate which category of objects we are interested in")
    parser.add_argument('--clustering', default='pyclustering', choices=('pyclustering', 'pyclust', 'local'))

    return parser.parse_args()


# def main():
#     args = arguments()
#
#     bboxes = get_class_data(cls=args.cls, dataset_path=args.dataset_path)
#
#     clustering = compute_kmedoids(bboxes, args.cls, option=args.clustering)
#
#     cluster_file = Path(args.dataset_path, 'clustering.jbl')
#
#     joblib.dump(clustering, cluster_file, compress=5)
#
#     ## For visualization
#     # clusters = joblib.load('clustering.jbl')
#     # draw_bboxes(clusters[25])
#     #
#     # for i in range(25, 36):
#     #     draw_bboxes(clusters[i])


# if __name__ == "__main__":
#     main()
