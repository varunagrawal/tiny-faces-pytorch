"""
Script to evaluate model.
Look at Makefile to see `evaluate` command.
"""

import argparse
import json

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import transforms

from tinyfaces.evaluation import get_detections, get_model


def arguments():
    parser = argparse.ArgumentParser("Image Evaluator")
    parser.add_argument("image_path")
    parser.add_argument("--checkpoint",
                        help="The path to the model checkpoint",
                        default="")
    parser.add_argument("--prob_thresh", type=float, default=0.6)
    parser.add_argument("--nms_thresh", type=float, default=0.3)

    return parser.parse_args()


def run(model, image, templates, prob_thresh, nms_thresh, device):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    img_transforms = transforms.Compose([transforms.ToTensor(), normalize])

    # Convert to tensor
    img = transforms.functional.to_tensor(image)

    rf = {'size': [859, 859], 'stride': [8, 8], 'offset': [-1, -1]}

    dets = get_detections(model,
                          img,
                          templates,
                          rf,
                          img_transforms,
                          prob_thresh,
                          nms_thresh,
                          scales=(0, ),
                          device=device)

    return dets


def main():
    args = arguments()

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    templates = json.load(open('tinyfaces/datasets/templates.json'))
    templates = np.round(np.array(templates), decimals=8)

    num_templates = templates.shape[0]

    model = get_model(args.checkpoint, num_templates=num_templates)
    print("Loaded model", args.checkpoint)

    image = Image.open(args.image_path).convert('RGB')

    with torch.no_grad():
        # run model on image
        dets = run(model, image, templates, args.prob_thresh, args.nms_thresh,
                   device)

    draw = ImageDraw.Draw(image)
    for det in dets:
        draw.rectangle(((det[0], det[1]), (det[2], det[3])))

    image.show()


if __name__ == "__main__":
    main()
