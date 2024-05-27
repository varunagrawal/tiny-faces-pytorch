from pathlib import Path

import numpy as np
import torch
from torchvision import transforms

from tinyfaces.datasets import get_dataloader
from tinyfaces.models.model import DetectionModel
from tinyfaces.models.utils import get_bboxes
from tinyfaces.utils.nms import nms


def val_dataloader(args):
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
                                                    int(min_side * scale))

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
    results_dir = results_dir or f"{split}_results"
    results_dir = Path(results_dir)

    if not results_dir.exists():
        results_dir.mkdir(parents=True)

    filename = results_dir / img_path.replace('jpg', 'txt')
    file_dir = filename.parent
    if not file_dir.exists():
        file_dir.mkdir(parents=True)

    with open(filename, 'w') as f:
        f.write(img_path.split('/')[-1] + "\n")
        f.write(str(dets.shape[0]) + "\n")

        for x in dets:
            left, top = np.round(x[0]), np.round(x[1])
            width = np.round(x[2] - x[0] + 1)
            height = np.round(x[3] - x[1] + 1)
            score = x[4]
            d = f"{int(left)} {int(top)} {int(width)} {int(height)} {score}\n"

            f.write(d)
