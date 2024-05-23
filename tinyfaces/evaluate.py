import argparse
import os
import os.path as osp

import numpy as np
import torch
from datasets import get_dataloader
from torchvision import transforms
from tqdm import tqdm

from tinyfaces.models.model import DetectionModel
from tinyfaces.models.utils import get_bboxes
from tinyfaces.utils.nms import nms


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


def get_model(checkpoint=None, num_templates=25):
    model = DetectionModel(num_templates=num_templates)
    if checkpoint:
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint["model"])
    return model


def get_detections(model,
                   img,
                   templates,
                   rf,
                   img_transforms,
                   prob_thresh=0.65,
                   nms_thresh=0.3,
                   scales=(-2, -1, 0, 1),
                   device=None):
    model = model.to(device)
    model.eval()

    dets = np.empty((0, 5))  # store bbox (x1, y1, x2, y2), score

    num_templates = templates.shape[0]

    # Evaluate over multiple scale
    scales_list = [2**x for x in scales]

    # convert tensor to PIL image so we can perform resizing
    image = transforms.functional.to_pil_image(img[0])

    min_side = np.min(image.size)

    for scale in scales_list:
        # scale the images
        scaled_image = transforms.functional.resize(image,
                                                    np.int(min_side * scale))

        # normalize the images
        img = img_transforms(scaled_image)

        # add batch dimension
        img.unsqueeze_(0)

        # now run the model
        x = img.float().to(device)

        output = model(x)

        # first `num_templates` channels are class maps
        score_cls = output[:, :num_templates, :, :]
        prob_cls = torch.sigmoid(score_cls)

        score_cls = score_cls.data.cpu().numpy().transpose((0, 2, 3, 1))
        prob_cls = prob_cls.data.cpu().numpy().transpose((0, 2, 3, 1))

        score_reg = output[:, num_templates:, :, :]
        score_reg = score_reg.data.cpu().numpy().transpose((0, 2, 3, 1))

        t_bboxes, scores = get_bboxes(score_cls, score_reg, prob_cls,
                                      templates, prob_thresh, rf, scale)

        scales = np.ones((t_bboxes.shape[0], 1)) / scale
        # append scores at the end for NMS
        d = np.hstack((t_bboxes, scores))

        dets = np.vstack((dets, d))

    # Apply NMS
    keep = nms(dets, nms_thresh)
    dets = dets[keep]

    return dets


def write_results(dets, img_path, split, results_dir=None):
    results_dir = results_dir or "{0}_results".format(split)

    if not osp.exists(results_dir):
        os.makedirs(results_dir)

    filename = osp.join(results_dir, img_path.replace('jpg', 'txt'))
    file_dir = os.path.dirname(filename)
    if not osp.exists(file_dir):
        os.makedirs(file_dir)

    with open(filename, 'w') as f:
        f.write(img_path.split('/')[-1] + "\n")
        f.write(str(dets.shape[0]) + "\n")

        for x in dets:
            left, top = np.round(x[0]), np.round(x[1])
            width = np.round(x[2] - x[0] + 1)
            height = np.round(x[3] - x[1] + 1)
            score = x[4]
            d = "{0} {1} {2} {3} {4}\n".format(int(left), int(top), int(width),
                                               int(height), score)
            f.write(d)


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
