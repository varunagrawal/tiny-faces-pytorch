"""Script to compute clusters of bounding boxes from the training set."""

import argparse
from pathlib import Path

import joblib
import numpy as np
from PIL import Image, ImageDraw

from tinyfaces.clustering.cluster import compute_kmedoids
from tinyfaces.datasets.wider_face import WIDERFace


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', type=Path)
    parser.add_argument(
        '--cls',
        default=1,
        type=int,
        help="Indicate which category of objects we are interested in")
    parser.add_argument('--clustering',
                        default='pyclustering',
                        choices=('pyclustering', 'pyclust', 'local'))

    return parser.parse_args()


def draw_bboxes(clusters):
    """
    Draw and save the clustered bounding boxes for inspection
    :param clusters:
    :return:
    """
    im = Image.new('RGB', [512, 512])
    d = ImageDraw.Draw(im)

    for bbox in clusters['medoids']:
        box = [(0, 0), (-bbox[0] + bbox[2], -bbox[1] + bbox[3])]
        color = tuple(np.random.choice(range(256), size=3))
        d.rectangle(box, outline=color)

    im.save("canonical_bbox_clusters_{0}.jpg".format(len(clusters['medoids'])))
    # im.show()


def main():
    args = arguments()

    dataset = WIDERFace(args.traindata.expanduser(), [])

    clustering = compute_kmedoids(dataset.get_all_bboxes(),
                                  args.cls,
                                  option=args.clustering)

    cluster_file = Path(args.dataset_path, 'clustering.jbl')

    joblib.dump(clustering, cluster_file, compress=5)

    # For visualization
    clusters = joblib.load('clustering.jbl')
    draw_bboxes(clusters[25])

    for i in range(25, 36):
        draw_bboxes(clusters[i])


if __name__ == "__main__":
    main()
