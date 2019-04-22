import numpy as np


def get_bboxes(score_cls, score_reg, templates, prob_thresh, rf, scale):
    """
    Convert model output tensor to a set of bounding boxes and their corresponding scores
    """
    num_templates = templates.shape[0]

    fb, fy, fx, fc = np.where(score_cls > prob_thresh)

    scores = score_cls[fb, fy, fx, fc]
    scores = scores.reshape((scores.shape[0], 1))

    stride, offset = rf['stride'], rf['offset']
    #TODO
    cy, cx = fy * stride[0] + offset[0], fx * stride[1] + offset[1]
    ch, cw = templates[fc, 3] - templates[fc, 1], \
        templates[fc, 2] - templates[fc, 0]

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
    bboxes = np.array(
        [rcx - rcw / 2, rcy - rch / 2, rcx + rcw / 2, rcy + rch / 2]).T

    # bboxes has a channel dim so we remove that
    bboxes = bboxes[0]

    # scale the bboxes
    bboxes = bboxes / scale

    return bboxes, scores
