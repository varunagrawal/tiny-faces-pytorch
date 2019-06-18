import numpy as np


def get_bboxes(score_cls, score_reg, templates, prob_thresh, rf, scale=1, refine=True):
    """
    Convert model output tensor to a set of bounding boxes and their corresponding scores
    """
    num_templates = templates.shape[0]

    # template to evaluate at every scale (Type A templates)
    all_scale_template_ids = np.arange(4, 12)

    # templates to evaluate at a single scale aka small scale (Type B templates)
    one_scale_template_ids = np.arange(18, 25)

    ignored_template_ids = np.setdiff1d(np.arange(25), np.concatenate((all_scale_template_ids,
                                                                       one_scale_template_ids)))

    template_scales = templates[:, 4]

    # if we down-sample, then we only need large templates
    if scale < 1:
        invalid_one_scale_idx = np.where(
            template_scales[one_scale_template_ids] >= 1.0)
    elif scale == 1:
        invalid_one_scale_idx = np.where(
            template_scales[one_scale_template_ids] != 1.0)
    elif scale > 1:
        invalid_one_scale_idx = np.where(
            template_scales[one_scale_template_ids] != 1.0)

    invalid_template_id = np.concatenate((ignored_template_ids,
                                          one_scale_template_ids[invalid_one_scale_idx]))

    # zero out prediction from templates that are invalid on this scale
    score_cls[:, :, invalid_template_id] = 0.0

    indices = np.where(score_cls > prob_thresh)
    fb, fy, fx, fc = indices

    scores = score_cls[fb, fy, fx, fc]
    scores = scores.reshape((scores.shape[0], 1))

    stride, offset = rf['stride'], rf['offset']
    cy, cx = fy * stride[0] + offset[0], fx * stride[1] + offset[1]
    cw = templates[fc, 2] - templates[fc, 0] + 1
    ch = templates[fc, 3] - templates[fc, 1] + 1

    # bounding box refinements
    tx = score_reg[:, :, :, 0:num_templates]
    ty = score_reg[:, :, :, 1 * num_templates:2 * num_templates]
    tw = score_reg[:, :, :, 2 * num_templates:3 * num_templates]
    th = score_reg[:, :, :, 3 * num_templates:4 * num_templates]

    if refine:
        bboxes = regression_refinement(tx, ty, tw, th,
                                       cx, cy, cw, ch,
                                       indices)

    else:
        bboxes = np.array([cx - cw/2, cy - ch/2, cx + cw/2, cy + ch/2])

    # bboxes has a channel dim so we remove that
    bboxes = bboxes[0]

    # scale the bboxes
    factor = 1 / scale
    bboxes = bboxes * factor

    return bboxes, scores


def regression_refinement(tx, ty, tw, th, cx, cy, cw, ch, indices):
    # refine the bounding boxes
    dcx = cw * tx[indices]
    dcy = ch * ty[indices]

    rcx = cx + dcx
    rcy = cy + dcy

    rcw = cw * np.exp(tw[indices])
    rch = ch * np.exp(th[indices])

    # create bbox array
    rcx = rcx.reshape((rcx.shape[0], 1))
    rcy = rcy.reshape((rcy.shape[0], 1))
    rcw = rcw.reshape((rcw.shape[0], 1))
    rch = rch.reshape((rch.shape[0], 1))

    # transpose so that it is (N, 4)
    bboxes = np.array(
        [rcx - rcw / 2, rcy - rch / 2, rcx + rcw / 2, rcy + rch / 2]).T

    return bboxes
