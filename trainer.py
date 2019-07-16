from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as nnfunc
from torchvision import transforms

from models.utils import get_bboxes
from utils.nms import nms


def print_state(idx, epoch, size, loss_cls, loss_reg):
    if epoch >= 0:
        message = "Epoch: [{0}][{1}/{2}]\t".format(epoch, idx, size)
    else:
        message = "Val: [{0}/{1}]\t".format(idx, size)

    print(message +
          '\tloss_cls: {loss_cls:.6f}' \
          '\tloss_reg: {loss_reg:.6f}'.format(loss_cls=loss_cls, loss_reg=loss_reg))


def save_checkpoint(state, filename="checkpoint.pth", save_path="weights"):
    # check if the save directory exists
    if not Path(save_path).exists():
        Path(save_path).mkdir()

    save_path = Path(save_path, filename)
    torch.save(state, str(save_path))


def visualize_output(img, output, templates, proc, prob_thresh=0.55, nms_thresh=0.1):
    tensor_to_image = transforms.ToPILImage()

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for t, m, s in zip(img[0], mean, std):
        t.mul_(s).add_(m)

    image = tensor_to_image(img[0])  # Index into the batch

    cls_map = nnfunc.sigmoid(output[:, 0:templates.shape[0], :, :]).data.cpu(
    ).numpy().transpose((0, 2, 3, 1))[0, :, :, :]
    reg_map = output[:, templates.shape[0]:, :, :].data.cpu(
    ).numpy().transpose((0, 2, 3, 1))[0, :, :, :]

    print(np.sort(np.unique(cls_map))[::-1])
    proc.visualize_heatmaps(image, cls_map, reg_map, templates,
                            prob_thresh=prob_thresh, nms_thresh=nms_thresh)

    p = input("Continue? [Yn]")
    if p.lower().strip() == 'n':
        exit(0)


def draw_bboxes(image, img_id, bboxes, scores, scales, processor):
    processor.render_and_save_bboxes(image, img_id, bboxes, scores, scales)


def train(model, loss_fn, optimizer, dataloader, epoch, device):
    model = model.to(device)
    model.train()

    for idx, (img, class_map, regression_map) in enumerate(dataloader):
        x = img.float().to(device)

        class_map_var = class_map.float().to(device)
        regression_map_var = regression_map.float().to(device)

        output = model(x)
        loss = loss_fn(output,
                       class_map_var, regression_map_var)

        # visualize_output(img, output, dataloader.dataset.templates)

        optimizer.zero_grad()
        # Get the gradients
        # torch will automatically mask the gradients to 0 where applicable!
        loss.backward()
        optimizer.step()

        print_state(idx, epoch, len(dataloader),
                    loss_fn.class_average.average,
                    loss_fn.reg_average.average)


def get_detections(model, img, templates, rf, img_transforms,
                   prob_thresh=0.65, nms_thresh=0.3, scales=(-2, -1, 0, 1), device=None):
    model = model.to(device)
    model.eval()

    dets = np.empty((0, 5))  # store bbox (x1, y1, x2, y2), score

    num_templates = templates.shape[0]

    # Evaluate over multiple scale
    scales_list = [2 ** x for x in scales]

    # convert tensor to PIL image so we can perform resizing
    image = transforms.functional.to_pil_image(img[0])

    min_side = np.min(image.size)

    for scale in scales_list:
        # scale the images
        scaled_image = transforms.functional.resize(image,
                                                    np.int(min_side*scale))

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
