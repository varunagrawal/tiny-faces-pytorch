"""Script to evaluate trained model."""

import argparse

import torch
from tqdm import tqdm

from tinyfaces.evaluation import (get_detections, get_model, val_dataloader,
                                  write_results)


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

    val_loader, templates = val_dataloader(args)
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
