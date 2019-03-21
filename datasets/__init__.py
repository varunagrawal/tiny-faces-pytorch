import numpy as np
import os
import os.path as osp
import json
from utils.cluster import compute_kmedoids
from .wider_face import WIDERFace
from torch.utils import data
from torchvision import transforms


def get_dataloader(datapath, args, num_templates=25, template_file="templates.json", train=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    img_transforms = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    data_loader = None

    template_file = osp.join("datasets", template_file)

    if osp.exists(template_file):
        templates = json.load(open(template_file))

    else:
        dataset = WIDERFace(osp.expanduser(args.traindata), [])
        clustering = compute_kmedoids(dataset.get_all_bboxes(), 1, indices=num_templates,
                                      option='pyclustering', max_clusters=num_templates)

        print("Canonical bounding boxes computed")
        templates = clustering[num_templates]['medoids'].tolist()
        
        # record templates
        json.dump(templates, open(template_file, "w"))

    templates = np.array(templates)
    
    data_loader = data.DataLoader(WIDERFace(osp.expanduser(datapath), templates,
                                            train=train, img_transforms=img_transforms,
                                            dataset_root=osp.expanduser(args.dataset_root)),
                                  batch_size=args.batch_size, shuffle=train,
                                  num_workers=args.workers, pin_memory=True)

    return data_loader
