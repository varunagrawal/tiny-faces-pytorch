"""
Script to evaluate model.
Look at Makefile to see `evaluate` command.
"""

import argparse

import torch
from torchvision import transforms
from tqdm import tqdm

from tinyfaces.datasets import get_dataloader
from tinyfaces.evaluate import get_detections, get_model, write_results


def arguments():
    parser = argparse.ArgumentParser("Model Evaluator")
    parser.add_argument("dataset")
    parser.add_argument("--split", default="val")
    parser.add_argument("--dataset-root")
    parser.add_argument("--checkpoint",
                        help="The path to the model checkpoint",
                        default="")
    parser.add_argument("--prob_thresh", type=float, default=0.03)
    parser.add_argument("--nms_thresh", type=float, default=0.3)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--results_dir", default=None)
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()


def dataloader(args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_transforms = transforms.Compose([transforms.ToTensor(), normalize])

    val_loader, templates = get_dataloader(args.dataset,
                                           args,
                                           train=False,
                                           split=args.split,
                                           img_transforms=val_transforms)
    return val_loader, templates


def run(model,
        val_loader,
        templates,
        prob_thresh,
        nms_thresh,
        device,
        split,
        results_dir=None,
        debug=False):
    for _, (img, filename) in tqdm(enumerate(val_loader),
                                   total=len(val_loader)):
        dets = get_detections(model,
                              img,
                              templates,
                              val_loader.dataset.rf,
                              val_loader.dataset.transforms,
                              prob_thresh,
                              nms_thresh,
                              device=device)

        write_results(dets, filename[0], split, results_dir)
    return dets


def main():
    args = arguments()

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    val_loader, templates = dataloader(args)
    num_templates = templates.shape[0]

    model = get_model(args.checkpoint, num_templates=num_templates)

    with torch.no_grad():
        # run model on val/test set and generate results files
        run(model,
            val_loader,
            templates,
            args.prob_thresh,
            args.nms_thresh,
            device,
            args.split,
            results_dir=args.results_dir,
            debug=args.debug)


if __name__ == "__main__":
    main()
