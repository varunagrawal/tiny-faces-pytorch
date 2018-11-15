import os
import os.path as osp
import numpy as np
import torch
import json
from sklearn.externals import joblib
from torch.utils import data
from torchvision import transforms
import trainer
import argparse
from datasets.coco import COCO
from models.models import DetectionModel
from pycocotools import coco
from pycocotools import cocoeval


def arguments():
    parser = argparse.ArgumentParser("Model Evaluator")
    parser.add_argument("valdata")
    parser.add_argument("--annots", help="Path to ground truth annotations")
    parser.add_argument("--dataset-root")
    parser.add_argument("--multiscale", action='store_true', default=False)
    parser.add_argument("--checkpoint", help="The path to the model checkpoint", default="")
    parser.add_argument("--prob_thresh", type=float, default=0.7)
    parser.add_argument("--nms_thresh", type=float, default=0.1)
    parser.add_argument("--workers", default=8, type=int)

    return parser.parse_args()


def dataloader(args):
    num_clusters = 25
    clusters = joblib.load("datasets/coco_clustering.jbl")[num_clusters]['medoids']
    clusters = np.array(clusters)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    val_loader = data.DataLoader(
        COCO(osp.expanduser(args.valdata), clusters, train=False, split="val", img_transforms=val_transforms,
             dataset_root=args.dataset_root, multiscale=args.multiscale),
        batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    return val_loader, clusters


def get_model(checkpoint=None, num_templates=25):
    model = DetectionModel(num_templates=num_templates)
    if checkpoint:
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint["model"])
    return model


def run(model, val_loader, clusters, prob_thresh, nms_thresh, predictions_file, multiscale=False):
    if osp.exists(predictions_file):
        os.remove(predictions_file)

    if multiscale:
        trainer.evaluate_multiscale(model, val_loader, clusters, prob_thresh=prob_thresh, nms_thresh=nms_thresh)
    else:
        trainer.evaluate(model, val_loader, clusters, prob_thresh=prob_thresh, nms_thresh=nms_thresh)


def evaluate(args):
    data_dir = args.dataset_root
    data_type = 'val2014'
    ann_file = "{0}/annotations/{1}_{2}.cars.json".format(data_dir, 'instances', data_type)
    results_file = "predictions.json"

    coco_gt = coco.COCO(ann_file)
    coco_det = coco_gt.loadRes(results_file)

    coco_eval = cocoeval.COCOeval(coco_gt, coco_det, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


def main():
    args = arguments()

    predictions = "predictions.json"
    # annots = args.valdata

    num_templates = 25

    val_loader, clusters = dataloader(args)
    model = get_model(args.checkpoint, num_templates=num_templates)

    run(model, val_loader, clusters, args.prob_thresh, args.nms_thresh, predictions, args.multiscale)

    # Use the COCO API to evaluate
    evaluate(args)


if __name__ == "__main__":
    main()
