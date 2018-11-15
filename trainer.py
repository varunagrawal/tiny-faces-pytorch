import numpy as np
import torch
from torchvision import transforms
from torch.nn import functional as nnfunc
from pathlib import Path
from tqdm import tqdm
from utils.nms import nms
import json
import flops


def print_state(idx, epoch, size, loss_cls, loss_reg):
    if epoch >= 0:
        message = "Epoch: [{0}][{1}/{2}]\t".format(epoch, idx, size)
    else:
        message = "Val: [{0}/{1}]\t".format(idx, size)

    print(message + '\tloss_cls: {loss_cls:.6f}\tloss_reg: {loss_reg:.6f}'.format(
        loss_cls=loss_cls, loss_reg=loss_reg))


def save_checkpoint(state, filename="checkpoint.pth.tar", save_path="weights"):
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


def train(model, loss_fn, optimizer, dataloader, epoch, save_path):
    model = model.train()
    model = model.cuda()

    for idx, (img, class_map, regression_map) in enumerate(dataloader):
        x = img.float().cuda()

        class_map_var = class_map.float().cuda()
        regression_map_var = regression_map.float().cuda()

        optimizer.zero_grad()

        output = model(x)

        # visualize_output(img, output, dataloader.dataset.templates)

        loss, cls_loss, reg_loss = loss_fn(
            output, class_map_var, regression_map_var)

        # Get the gradients
        # torch will automatically mask the gradients to 0 where applicable!
        loss.backward()

        optimizer.step()

        print_state(idx, epoch, len(dataloader),
                    loss_fn.cls_average.average,
                    loss_fn.reg_average.average)

    save_checkpoint({
        'epoch': epoch + 1,
        'batch_size': dataloader.batch_size,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }, filename="checkpoint_{0}.pth".format(epoch+1), save_path=save_path)


def evaluate_multiscale(model, dataloader, templates, prob_thresh=0.65, nms_thresh=0.3, num_templates=25):
    print("Running multiscale evaluation code")

    model = model.eval().cuda()

    # Multi scale stuff
    # scaling_factors = [0.25, 0.5, 1, 2]  # 2
    # scaling_factors = [0.25, 0.5, 1, 2, 4]  # 2
    # scaling_factors = [1]  # 0
    # scaling_factors = [0.25, 0.5, 0.7, 0.9, 1, 1.1, 1.5, 2]  # 4
    scales_list = [0.7 ** x for x in [4, 3, 2, 1, 0, -1]]

    results = []
    to_pil_image = transforms.ToPILImage()

    for idx, (img, image_id, labels) in tqdm(enumerate(dataloader), total=len(dataloader)):
        dets = np.empty((0, 6))  # store bbox (x1, y1, x2, y2), score and scale

        # convert tensor to PIL image so we can perform resizing
        image = to_pil_image(img[0])

        min_side = np.min(image.size)

        for s, scale in enumerate(scales_list):
            # scale the images
            scaled_image = transforms.Resize(np.int(min_side*scale))(image)

            # normalize the images
            img = dataloader.dataset.transforms(scaled_image)
            # add batch dimension
            img.unsqueeze_(0)

            # now run the model
            x = img.float().cuda()

            output = model(x)

            # first `num_templates` channels are class maps
            score_cls = torch.sigmoid(output[:, :num_templates, :, :])
            score_cls = score_cls.data.cpu().numpy().transpose((0, 2, 3, 1))

            score_reg = output[:, num_templates:, :, :]
            score_reg = score_reg.data.cpu().numpy().transpose((0, 2, 3, 1))

            fb, fy, fx, fc = np.where(score_cls > prob_thresh)

            scores = score_cls[fb, fy, fx, fc]
            scores = scores.reshape((scores.shape[0], 1))

            rf = dataloader.dataset.rf
            strx, offset = rf['stride'], rf['offset']
            cy, cx = fy * strx[0] + offset[0], fx * strx[1] + offset[1]
            ch, cw = templates[fc, 3] - templates[fc, 1] + 1, templates[fc, 2] - templates[fc, 0] + 1

            # bounding box refinements
            tx = score_reg[:, :, :, 0:num_templates]
            ty = score_reg[:, :, :, 1 * num_templates:2 * num_templates]
            tw = score_reg[:, :, :, 2 * num_templates:3 * num_templates]
            th = score_reg[:, :, :, 3 * num_templates:4 * num_templates]

            # refine the bounding boxes
            dcx = cw * tx[fb, fy, fx, fc]
            dcy = ch * ty[fb, fy, fx, fc]

            rcx = cx + dcx
            rcy = cy + dcy

            rcw = cw * np.exp(tw[fb, fy, fx, fc])
            rch = ch * np.exp(th[fb, fy, fx, fc])

            # create bbox array and scale the coords
            rcx = rcx.reshape((rcx.shape[0], 1))
            rcy = rcy.reshape((rcy.shape[0], 1))
            rcw = rcw.reshape((rcw.shape[0], 1))
            rch = rch.reshape((rch.shape[0], 1))

            # transpose so that it is (N, 4)
            t_bboxes = np.array(
                [rcx - rcw / 2, rcy - rch / 2, rcx + rcw / 2, rcy + rch / 2]).T

            # bboxes has a channel dim
            t_bboxes = t_bboxes[0]

            # scale the bboxes
            t_bboxes = t_bboxes / scale

            scales = np.ones((t_bboxes.shape[0], 1)) / scale
            # append scores at the end for NMS
            d = np.hstack((t_bboxes, scores, scales))

            dets = np.vstack((dets, d))

        # Apply NMS
        keep = nms(dets, nms_thresh)
        dets = dets[keep]

        # draw_bboxes(imgs[i], "{0}_{1}".format(image_id.item(), i), d[:, 0:4], dataloader.dataset.processor)
        # draw_bboxes(image, "{0}".format(image_id.item()),
        #             dets[:, 0:4], dets[:, 4], 1/dets[:, 5], dataloader.dataset.processor)

        # Save to COCO Evaluation format
        bb = dets[:, 0:4]
        bb[:, 2] = bb[:, 2] - bb[:, 0] + 1
        bb[:, 3] = bb[:, 3] - bb[:, 1] + 1

        for ind in range(bb.shape[0]):
            results.append({
                "image_id": image_id[0].item(),
                "category_id": 3,
                "bbox": bb[ind, :].tolist(),
                "score": dets[ind, 4].tolist()
            })

    with open("predictions.json", "w") as pred_file:
        json.dump(results, pred_file)

    print("Results saved to `predictions.json`")


def evaluate(model, dataloader, templates, prob_thresh, nms_thresh, num_templates=25, debug=False):
    print("Running evaluation code")
    model.eval()
    model.cuda()

    results = []

    for idx, (img, image_id, labels) in tqdm(enumerate(dataloader), total=len(dataloader)):
        x = img.float().cuda()
        output = model(x)

        # first n_templates channels are class maps
        score_cls = torch.sigmoid(
            output[:, :num_templates, :, :]).data.cpu().numpy().transpose((0, 2, 3, 1))
        score_reg = output[:, num_templates:, :,
                           :].data.cpu().numpy().transpose((0, 2, 3, 1))
        # print(score_reg.max(), score_reg.min())

        if debug:
            visualize_output(img, output, dataloader.dataset.templates,
                             dataloader.dataset.processor, prob_thresh, nms_thresh)

        fb, fy, fx, fc = np.where(score_cls > prob_thresh)

        scores = score_cls[fb, fy, fx, fc]
        scores = scores.reshape((scores.shape[0], 1))

        rf = dataloader.dataset.rf
        strx, offset = rf['stride'], rf['offset']
        cy, cx = fy*strx[0] + offset[0], fx*strx[1] + offset[1]
        ch, cw = templates[fc, 3] - templates[fc, 1] + \
            1, templates[fc, 2] - templates[fc, 0] + 1

        # bounding box refinements
        tx = score_reg[:, :, :, 0:num_templates]
        ty = score_reg[:, :, :, 1 * num_templates:2 * num_templates]
        tw = score_reg[:, :, :, 2 * num_templates:3 * num_templates]
        th = score_reg[:, :, :, 3 * num_templates:4 * num_templates]

        # refine the bounding boxes
        dcx = cw * tx[fb, fy, fx, fc]
        dcy = ch * ty[fb, fy, fx, fc]

        rcx = cx + dcx
        rcy = cy + dcy

        rcw = cw * np.exp(tw[fb, fy, fx, fc])
        rch = ch * np.exp(th[fb, fy, fx, fc])

        # create bbox array and scale the coords
        rcx = rcx.reshape((rcx.shape[0], 1))
        rcy = rcy.reshape((rcy.shape[0], 1))
        rcw = rcw.reshape((rcw.shape[0], 1))
        rch = rch.reshape((rch.shape[0], 1))

        # transpose so that it is (1, N, 4) not (4, N, 1)
        t_bboxes = np.array([rcx-rcw/2, rcy-rch/2, rcx+rcw/2, rcy+rch/2]).T

        t_bboxes = t_bboxes[0]

        # append scores at the end for NMS
        dets = np.hstack((t_bboxes, scores))

        keep = nms(dets, nms_thresh)
        dets = dets[keep]

        # draw_bboxes(img, image_id, dets[:, 0:4], dataloader.dataset.processor)

        # Save to COCO Evaluation format
        bb = dets[:, 0:4]
        bb[:, 2] = bb[:, 2] - bb[:, 0] + 1
        bb[:, 3] = bb[:, 3] - bb[:, 1] + 1

        for ind in range(bb.shape[0]):
            results.append({
                "image_id": image_id[0].item(),
                "category_id": 3,
                "bbox": bb[ind, :].tolist(),
                "score": dets[ind, 4].tolist()
            })

    with open("predictions.json", "w") as pred_file:
        json.dump(results, pred_file)

    print("Results saved to `predictions.json`")
