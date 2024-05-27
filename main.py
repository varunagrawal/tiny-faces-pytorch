"""
Main model training training script.
See Makefile `main` to see usage.
"""
import argparse
from pathlib import Path

import torch
from torch import optim
from torchvision import transforms

from tinyfaces import trainer
from tinyfaces.datasets import get_dataloader
from tinyfaces.models.loss import DetectionCriterion
from tinyfaces.models.model import DetectionModel


def arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("traindata")
    parser.add_argument("valdata")
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--dataset", default="WIDERFace")
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight-decay", default=0.0005, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--batch_size", default=12, type=int)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--start-epoch", default=0, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--save-every", default=10, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()


def main():
    args = arguments()

    num_templates = 25  # aka the number of clusters

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    img_transforms = transforms.Compose([transforms.ToTensor(), normalize])
    train_loader, _ = get_dataloader(args.traindata,
                                     args,
                                     num_templates,
                                     img_transforms=img_transforms)

    model = DetectionModel(num_objects=1, num_templates=num_templates)
    loss_fn = DetectionCriterion(num_templates)

    # directory where we'll store model weights
    weights_dir = Path("weights")
    if not weights_dir.exists():
        weights_dir.mkdir()

    # check for CUDA
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    # As per Peiyun, SGD is more robust than Adam and works really well
    optimizer = optim.SGD(model.learnable_parameters(args.lr),
                          lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # Set the start epoch if it has not been
        if not args.start_epoch:
            args.start_epoch = checkpoint['epoch']

    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=20,
                                          last_epoch=args.start_epoch - 1)

    # train and evalute for `epochs`
    for epoch in range(args.start_epoch, args.epochs):
        trainer.train(model,
                      loss_fn,
                      optimizer,
                      train_loader,
                      epoch,
                      device=device)
        scheduler.step()

        if (epoch + 1) % args.save_every == 0:
            trainer.save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'batch_size': train_loader.batch_size,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                },
                filename="checkpoint_{0}.pth".format(epoch + 1),
                save_path=weights_dir)


if __name__ == '__main__':
    main()
