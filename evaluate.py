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
from datasets import get_dataloader
from datasets.wider_face import WIDERFace
from models.model import DetectionModel


def arguments():
    parser = argparse.ArgumentParser("Model Evaluator")
    parser.add_argument("testdata")
    parser.add_argument("--dataset-root")
    parser.add_argument("--checkpoint",
                        help="The path to the model checkpoint", default="")
    parser.add_argument("--prob_thresh", type=float, default=0.7)
    parser.add_argument("--nms_thresh", type=float, default=0.1)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--batch_size", default=1)

    return parser.parse_args()


def dataloader(args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    val_loader, templates = get_dataloader(args.testdata, args,
                                           train=False, split="test", img_transforms=val_transforms)
    return val_loader, templates


def get_model(checkpoint=None, num_templates=25):
    model = DetectionModel(num_templates=num_templates)
    if checkpoint:
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint["model"])
    return model


def run(model, val_loader, templates, prob_thresh, nms_thresh, device):
    dets = trainer.evaluate(model, val_loader, templates,
                            prob_thresh, nms_thresh, device)
    return dets


def main():
    args = arguments()

    predictions_file = "predictions.json"

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    val_loader, templates = dataloader(args)
    num_templates = templates.shape[0]

    model = get_model(args.checkpoint, num_templates=num_templates)

    detections = run(model, val_loader, templates, args.prob_thresh,
                     args.nms_thresh, device)

    #TODO save results as per WIDERFace format

if __name__ == "__main__":
    main()
