import json
from pathlib import Path

import numpy as np
from torch.utils import data

from tinyfaces.clustering.cluster import compute_kmedoids
from tinyfaces.datasets.wider_face import WIDERFace

from utils.cluster import compute_kmedoids

from .wider_face import WIDERFace


def get_dataloader(datapath,
                   args,
                   num_templates=25,
                   template_file="templates.json",
                   img_transforms=None,
                   train=True,
                   split="train"):
    template_file = Path(__file__).parent / template_file

    if template_file.exists():
        templates = json.load(open(template_file))

    else:
        # Cluster the bounding boxes to get the templates
        dataset = WIDERFace(Path(args.traindata).expanduser(), [])
        clustering = compute_kmedoids(dataset.get_all_bboxes(),
                                      1,
                                      indices=num_templates,
                                      option='pyclustering',
                                      max_clusters=num_templates)

        print("Canonical bounding boxes computed")
        templates = clustering[num_templates]['medoids'].tolist()

        # record templates
        json.dump(templates, open(template_file, "w"))

    templates = np.round(np.array(templates), decimals=8)

    dataset = WIDERFace(Path(datapath).expanduser(),
                        templates,
                        split=split,
                        img_transforms=img_transforms,
                        dataset_root=Path(args.dataset_root).expanduser(),
                        debug=args.debug)
    data_loader = data.DataLoader(dataset,
                                  batch_size=args.batch_size,
                                  shuffle=train,
                                  num_workers=args.workers,
                                  pin_memory=True)

    return data_loader, templates
