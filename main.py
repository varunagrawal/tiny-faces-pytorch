import argparse
import torch
from torch import optim
from models.model import DetectionModel
from models.loss import DetectionCriterion
from datasets import get_dataloader
import trainer

import os, os.path as osp


def arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("traindata")
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--dataset", default="COCO", choices=('COCO', 'WIDERFace'))
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight-decay", default=0.0005, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--batch-size", default=12, type=int)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--start-epoch", default=0, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--resume", default="")

    return parser.parse_args()


def main():
    args = arguments()

    num_templates = 25  # aka the number of clusters

    train_loader = get_dataloader(args.traindata, args, num_templates)
    
    model = DetectionModel(num_objects=1, num_templates=num_templates)
    loss_fn = DetectionCriterion(num_templates)

    # directory where we'll store model weights
    weights_dir = "weights"
    if not osp.exists(weights_dir):
        os.mkdir(weights_dir)

    optimizer = optim.SGD(model.learnable_parameters(args.lr), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # Set the start epoch if it has not been
        if not args.start_epoch:
            args.start_epoch = checkpoint['epoch']

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, last_epoch=args.start_epoch-1)

    # train and evalute for `epochs`
    for epoch in range(args.start_epoch, args.epochs):
        scheduler.step()
        trainer.train(model, loss_fn, optimizer, train_loader, epoch, save_path=weights_dir)


if __name__ == '__main__':
    main()
