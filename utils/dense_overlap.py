import numpy as np


def compute_dense_overlap(ofx, ofy, stx, sty, vsx, vsy,
                          dx1, dy1, dx2, dy2,
                          gx1, gy1, gx2, gy2, zmx=1, zmy=1):
    """
    Compute the dense IoU
    """
    num_templates = dx1.shape[0]
    num_gt = gx1.shape[0]

    ty, tx = (vsy - 1) * zmy + 1, (vsx - 1) * zmx + 1  # + 1 is by definition of receptive field
    overlap = np.zeros((ty, tx, num_templates, num_gt))

    for i in range(num_gt):
        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = gx1[i], gy1[i], gx2[i], gy2[i]
        bbox_w, bbox_h = bbox_x2 - bbox_x1 + 1, bbox_y2 - bbox_y1 + 1
        bbox_area = bbox_w * bbox_h

        for j in range(num_templates):
            delta_x1, delta_y1, delta_x2, delta_y2 = dx1[j], dy1[j], dx2[j], dy2[j]
            filter_h = delta_y2 - delta_y1 + 1
            filter_w = delta_x2 - delta_x1 + 1

            filter_area = filter_w * filter_h

            xmax = tx
            ymax = ty

            # enumerate spatial locations
            for x in range(xmax):
                for y in range(ymax):
                    cx = ofx + x*(stx/zmx)
                    cy = ofy + y*(sty/zmy)

                    x1 = delta_x1 + cx
                    y1 = delta_y1 + cy
                    x2 = delta_x2 + cx
                    y2 = delta_y2 + cy

                    xx1 = max(x1, bbox_x1)
                    yy1 = max(y1, bbox_y1)
                    xx2 = min(x2, bbox_x2)
                    yy2 = min(y2, bbox_y2)

                    int_w = xx2 - xx1 + 1
                    int_h = yy2 - yy1 + 1

                    if int_h > 0 and int_w > 0:
                        int_area = int_w * int_h
                        union_area = filter_area + bbox_area - int_area

                        overlap[y, x, j, i] = int_area / union_area

                    else:
                        overlap[y, x, j, i] = 0

    # truncate the number of decimals to match MATLAB behavior
    return np.around(overlap, decimals=14)
