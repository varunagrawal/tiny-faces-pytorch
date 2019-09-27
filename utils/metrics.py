import json
import warnings

import numpy as np
from tqdm import tqdm


def jaccard_index(box_a, box_b, indices=[]):
    """
    Compute the Jaccard Index (Intersection over Union) of 2 boxes. Each box is (x1, y1, x2, y2).
    :param box_a:
    :param box_b:
    :param indices: The indices of box_a and box_b as [box_a_idx, box_b_idx]. Helps in debugging DivideByZero errors
    :return:
    """
    # area of bounding boxes
    area_A = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_B = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    intersection = (xB - xA) * (yB - yA)
    union = area_A + area_B - intersection

    # return the intersection over union value
    try:
        if union <= 0:
            iou = 0
        else:
            iou = intersection / union
    except:
        print(indices)
        print(box_a)
        print(box_b)
        print(area_A, area_B, intersection)
        exit(1)

    return iou


def rect_dist(I, J):
    if len(I.shape) == 1:
        I = I[np.newaxis, :]
        J = J[np.newaxis, :]

    # area of boxes
    aI = (I[:, 2] - I[:, 0] + 1) * (I[:, 3] - I[:, 1] + 1)
    aJ = (J[:, 2] - J[:, 0] + 1) * (J[:, 3] - J[:, 1] + 1)

    x1 = np.maximum(I[:, 0], J[:, 0])
    y1 = np.maximum(I[:, 1], J[:, 1])
    x2 = np.minimum(I[:, 2], J[:, 2])
    y2 = np.minimum(I[:, 3], J[:, 3])

    aIJ = (x2 - x1 + 1) * (y2 - y1 + 1) * (np.logical_and(x2 > x1, y2 > y1))

    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            iou = aIJ / (aI + aJ - aIJ)
        except (RuntimeWarning, Exception):
            iou = np.zeros(aIJ.shape)

    # set NaN, inf, and -inf to 0
    iou[np.isnan(iou)] = 0
    iou[np.isinf(iou)] = 0

    dist = np.maximum(np.zeros(iou.shape), np.minimum(np.ones(iou.shape), 1 - iou))

    return dist


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec)
    Compute VOC AP given precision and recall.
    Always uses the newer metric (in contrast to the '07 metric)
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def average_precision(confidence, dets, image_ids, class_recs, npos, ovthresh=0.5):
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = dets[sorted_ind, :]
    img_ids = [image_ids[x] for x in sorted_ind]

    nd = len(img_ids)  # num of detections
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    for d in tqdm(range(nd), total=nd):
        R = class_recs[img_ids[d]]
        bb = BB[d, :].astype(np.float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(np.float)
        BBGT[:, 2] = BBGT[:, 0] + BBGT[:, 2] - 1
        BBGT[:, 3] = BBGT[:, 1] + BBGT[:, 3] - 1

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin, 0.0)
            ih = np.maximum(iymax - iymin, 0.0)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                   (BBGT[:, 2] - BBGT[:, 0]) * (BBGT[:, 3] - BBGT[:, 1]) -
                   inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['det'][jmax]:
                tp[d] = 1.
                R['det'][jmax] = 1
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.


    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec)

    return ap, prec, rec


def compute_model_score(pred_file, gt_file, class_id=3):
    # load GT
    GT = json.load(open(gt_file))
    recs = {}
    for g in GT:
        recs[g["image"]["id"]] = g["bboxes"]

    class_recs = {}
    npos = 0
    for img_id in recs.keys():
        # get the list of all bboxes belonging to the particular class
        R = [obj for obj in recs[img_id] if obj["category_id"] == class_id]
        bboxes = np.array([x["bbox"] for x in R])
        det = [False] * len(R)  # to record if this object has already been recorded
        npos = npos + len(R)
        class_recs[img_id] = {
            'bbox': bboxes,
            'det': det
        }

    print("Loaded GT")

    # Read the detections
    with open(pred_file) as f:
        preds = f.readlines()
    preds = [json.loads(x) for x in preds]

    confidence, BB, image_ids = [], [], []
    for x in tqdm(preds, total=len(preds)):
        confidence.extend(x["confidences"])
        BB.extend(x["bboxes"])
        image_ids.extend([x["id"]]*len(x['confidences']))

    print("Loaded detections")

    confidence = np.array(confidence)
    BB = np.array(BB)

    print(confidence.shape)
    print(BB.shape)

    ap, prec, rec = average_precision(confidence, BB, image_ids, class_recs, npos)
    return ap, prec, rec
