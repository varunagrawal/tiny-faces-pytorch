import numpy as np
import os
import os.path as osp
from sklearn.externals import joblib
from utils.cluster import compute_kmedoids
from .coco import COCO
from .wider_face import WIDERFace
from torch.utils import data
from torchvision import transforms


def get_dataloader(datapath, args, num_clusters=25, train=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    img_transforms = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    data_loader = None

    # directory where we'll store model weights and cluster etc
    data_dir = "{0}_weights".format(args.dataset.lower())
    if not osp.exists(data_dir):
        os.mkdir(data_dir)

    dataset = WIDERFace(osp.expanduser(args.traindata), [])

    cluster_file = osp.join(data_dir, "clustering.jbl")
    if osp.exists(cluster_file):
        clusters = joblib.load(cluster_file)[num_clusters]['medoids']
    else:
        clustering = compute_kmedoids(dataset.get_all_bboxes(), 1, indices=num_clusters,
                                      option='pyclustering', max_clusters=num_clusters)
        print("Canonical bounding boxes computed")
        clusters = clustering[num_clusters]['medoids']
        joblib.dump(clustering, cluster_file, compress=5)

    # print(clusters)

    data_loader = data.DataLoader(
        WIDERFace(osp.expanduser(datapath), clusters, train=train, img_transforms=img_transforms,
                  dataset_root=osp.expanduser(args.dataset_root)),
        batch_size=args.batch_size, shuffle=train,
        num_workers=args.workers, pin_memory=True)

    return data_loader, data_dir
