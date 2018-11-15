import numpy as np
from copy import deepcopy

from utils.visualize import draw_bounding_box
from utils.nms import nms
from utils.metrics import rect_dist
from utils.dense_overlap import compute_dense_overlap
import logging


logger = logging.getLogger("detector")


class DataProcessor:
    """
    This is a helper class to abstract out all the operation needed during the data-loading pipeline of the Tiny Faces
    object detector.

    The idea is that this can act as a mixin that enables torch dataloaders with the heatmap generation semantics.
    """
    def __init__(self, input_size, heatmap_size, pos_thresh, neg_thresh, clusters, img_means=None, rf=None):
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.pos_thresh = pos_thresh
        self.neg_thresh = neg_thresh
        self.clusters = clusters
        self.rf = rf
        self.ofy, self.ofx = rf['offset']
        self.sty, self.stx = rf['stride']
        self.sample_size = 256
        self.img_means = img_means or [0.485, 0.456, 0.406]

    def balance_sampling(self, label_cls, pos_fraction):
        """
        Perform balance sampling by always sampling `pos_fraction` positive samples and
        `(1-pos_fraction)` negative samples from the input
        :param label_cls:
        :param pos_fraction:
        :return:
        """
        pos_maxnum = self.sample_size * pos_fraction  # sample 128 positive points
        # find all the points where we have objects
        # ravel the indices to get a 1D array. This makes the subsequent operations easier to reason about
        pos_idx_unraveled = np.where(label_cls == 1)
        pos_idx = np.ravel_multi_index(pos_idx_unraveled, label_cls.shape)

        if pos_idx.size > pos_maxnum:
            # Get all the indices of the locations to be zeroed out
            didx = self.shuffle_index(pos_idx.size, pos_idx.size-pos_maxnum)
            # Get the locations and unravel it so we can index
            pos_idx_unraveled = np.unravel_index(pos_idx[didx], label_cls.shape)
            label_cls[pos_idx_unraveled] = 0

        neg_maxnum = pos_maxnum * (1 - pos_fraction) / pos_fraction
        neg_idx_unraveled = np.where(label_cls == -1)
        neg_idx = np.ravel_multi_index(neg_idx_unraveled, label_cls.shape)

        if neg_idx.size > neg_maxnum:
            # Get all the indices of the locations to be zeroed out
            ridx = self.shuffle_index(neg_idx.size, neg_maxnum)
            didx = np.arange(0, neg_idx.size)
            didx = np.delete(didx, ridx)
            neg_idx = np.unravel_index(neg_idx[didx], label_cls.shape)
            label_cls[neg_idx] = 0

        return label_cls

    def shuffle_index(self, n, n_out):
        """
        Randomly shuffle the indices and return a subset of them
        :param n: The number of indices to shuffle.
        :param n_out: The number of output indices.
        :return:
        """
        n = int(n)
        n_out = int(n_out)

        if n == 0 or n_out == 0:
            return np.empty(0)

        x = np.random.permutation(n)

        # the output should be at most the size of the input
        assert n_out <= n

        if n_out != n:
            x = x[:n_out]

        return x

    def crop_image(self, img, bboxes):
        """
        Crop a 500x500 patch from the image, taking care for smaller images.
        bboxes is the np.array of all bounding boxes [x1, y1, x2, y2]
        """
        # randomly pick a cropping window for the image
        crop_x1 = np.random.randint(0, np.max([1, img.shape[1] - self.input_size[1]]))
        crop_y1 = np.random.randint(0, np.max([1, img.shape[0] - self.input_size[0]]))
        crop_x2 = min(img.shape[1]-1, crop_x1 + self.input_size[1] - 1)
        crop_y2 = min(img.shape[0]-1, crop_y1 + self.input_size[0] - 1)
        crop_h = crop_y2 - crop_y1 + 1
        crop_w = crop_x2 - crop_x1 + 1

        # place the cropped image in a random location in a `input_size` image
        paste_box = [0, 0, 0, 0]  # x1, y1, x2, y2
        paste_box[0] = np.random.randint(0, self.input_size[1] - crop_w + 1)
        paste_box[1] = np.random.randint(0, self.input_size[0] - crop_h + 1)
        paste_box[2] = paste_box[0] + crop_w - 1
        paste_box[3] = paste_box[1] + crop_h - 1

        # set this to average image colors
        # this will later be subtracted in mean image subtraction
        img_buf = np.zeros((self.input_size + (3,)))

        # add the average image so it gets subtracted later.
        for i, c in enumerate(self.img_means):
            # img is a int8 array, so we need to scale the values accordingly
            img_buf[:, :, i] = int(c*255)

        img_buf[paste_box[1]:paste_box[3], paste_box[0]:paste_box[2], :] = img[crop_y1:crop_y2, crop_x1:crop_x2, :]

        if bboxes.shape[0] > 0:
            # check if overlap is above negative threshold
            tbox = deepcopy(bboxes)
            tbox[:, 0] = np.maximum(tbox[:, 0], crop_x1)
            tbox[:, 1] = np.maximum(tbox[:, 1], crop_y1)
            tbox[:, 2] = np.minimum(tbox[:, 2], crop_x2)
            tbox[:, 3] = np.minimum(tbox[:, 3], crop_y2)

            overlap = 1 - rect_dist(tbox, bboxes)

            # adjust the bounding boxes - first for crop and then for random placement
            bboxes[:, 0] = bboxes[:, 0] - crop_x1 + paste_box[0]
            bboxes[:, 1] = bboxes[:, 1] - crop_y1 + paste_box[1]
            bboxes[:, 2] = bboxes[:, 2] - crop_x1 + paste_box[0]
            bboxes[:, 3] = bboxes[:, 3] - crop_y1 + paste_box[1]

            # correct for bbox to be within image border
            bboxes[:, 0] = np.minimum(self.input_size[1]-1, np.maximum(0, bboxes[:, 0]))
            bboxes[:, 1] = np.minimum(self.input_size[0]-1, np.maximum(0, bboxes[:, 1]))
            bboxes[:, 2] = np.minimum(self.input_size[1]-1, np.maximum(1, bboxes[:, 2]))
            bboxes[:, 3] = np.minimum(self.input_size[0]-1, np.maximum(1, bboxes[:, 3]))

            # check to see if the adjusted bounding box is invalid
            invalid = np.logical_or(np.logical_or(bboxes[:, 2] <= bboxes[:, 0], bboxes[:, 3] <= bboxes[:, 1]),
                                    overlap < self.neg_thresh)

            # remove invalid bounding boxes
            ind = np.where(invalid)
            bboxes = np.delete(bboxes, ind, 0)

        return img_buf, bboxes, paste_box

    def get_padding(self, paste_box):
        """
        Get the padding of the image based on where the sampled image patch was placed.
        :param paste_box: [x1, y1, x2, y2]
        :return:
        """
        ofx, ofy = self.rf['offset']
        stx, sty = self.rf['stride']
        vsy, vsx = self.heatmap_size
        coarse_x, coarse_y = np.meshgrid(ofx + np.array(range(vsx)) * stx,
                                         ofy + np.array(range(vsy)) * sty)

        # each cluster is [x1, y1, x2, y2]
        dx1 = self.clusters[:, 0]
        dy1 = self.clusters[:, 1]
        dx2 = self.clusters[:, 2]
        dy2 = self.clusters[:, 3]

        # compute the bounds
        # We add new axes so that the arrays are numpy broadcasting compatible
        coarse_xx1 = coarse_x[:, :, np.newaxis] + dx1[np.newaxis, np.newaxis, :]  # (vsy, vsx, nt)
        coarse_yy1 = coarse_y[:, :, np.newaxis] + dy1[np.newaxis, np.newaxis, :]  # (vsy, vsx, nt)
        coarse_xx2 = coarse_x[:, :, np.newaxis] + dx2[np.newaxis, np.newaxis, :]  # (vsy, vsx, nt)
        coarse_yy2 = coarse_y[:, :, np.newaxis] + dy2[np.newaxis, np.newaxis, :]  # (vsy, vsx, nt)

        padx1 = coarse_xx1 < paste_box[0]
        pady1 = coarse_yy1 < paste_box[1]
        padx2 = coarse_xx2 > paste_box[2]
        pady2 = coarse_yy2 > paste_box[3]

        pad_mask = padx1 | pady1 | padx2 | pady2

        return pad_mask

    def get_regression(self, bboxes, cluster_boxes, iou):
        """
        Compute the target bounding box regression values
        :param bboxes:
        :param cluster_boxes:
        :param iou:
        :return:
        """
        ofx, ofy = self.rf['offset']
        stx, sty = self.rf['stride']
        vsy, vsx = self.heatmap_size

        coarse_xx, coarse_yy = np.meshgrid(ofx + np.array(range(vsx)) * stx,
                                           ofy + np.array(range(vsy)) * sty)

        dx1, dy1, dx2, dy2 = cluster_boxes

        # We reshape to take advantage of numpy broadcasting
        fxx1 = bboxes[:, 0].reshape(1, 1, 1, bboxes.shape[0])  # (1, 1, 1, bboxes)
        fyy1 = bboxes[:, 1].reshape(1, 1, 1, bboxes.shape[0])
        fxx2 = bboxes[:, 2].reshape(1, 1, 1, bboxes.shape[0])
        fyy2 = bboxes[:, 3].reshape(1, 1, 1, bboxes.shape[0])

        h = dy2 - dy1 + 1
        dhh = h.reshape(1, 1, h.shape[0], 1)  # (1, 1, N, 1)
        w = dx2 - dx1 + 1
        dww = w.reshape(1, 1, w.shape[0], 1)  # (1, 1, N, 1)

        fcx = (fxx1 + fxx2) / 2
        fcy = (fyy1 + fyy2) / 2

        tx = np.divide((fcx - coarse_xx.reshape(vsy, vsx, 1, 1)), dww)
        ty = np.divide((fcy - coarse_yy.reshape(vsy, vsx, 1, 1)), dhh)

        fhh = fyy2 - fyy1 + 1
        fww = fxx2 - fxx1 + 1

        tw = np.log(np.divide(fww, dww))  # (1, 1, N, bboxes)
        th = np.log(np.divide(fhh, dhh))

        # Randomly perturb the IOU so that if multiple candidates have the same IOU,
        # we don't pick the same one every time. This is useful when the template is smaller than the GT bbox
        iou = iou + (1e-6 * np.random.rand(*iou.shape))

        best_obj_per_loc = iou.argmax(axis=3)
        idx0, idx1, idx2 = np.indices(iou.shape[:-1])

        tx = tx[idx0, idx1, idx2, best_obj_per_loc]
        ty = ty[idx0, idx1, idx2, best_obj_per_loc]

        tw = np.repeat(tw, vsy, axis=0)  # (vsy, 1, N, bboxes)
        tw = np.repeat(tw, vsx, axis=1)  # (vsy, vsx, N, bboxes)
        tw = tw[idx0, idx1, idx2, best_obj_per_loc]

        th = np.repeat(th, vsy, axis=0)
        th = np.repeat(th, vsx, axis=1)
        th = th[idx0, idx1, idx2, best_obj_per_loc]

        return np.concatenate((tx, ty, tw, th), axis=2), iou

    def get_heatmaps(self, bboxes, pad_mask):
        ofx, ofy = self.rf['offset']
        stx, sty = self.rf['stride']
        vsy, vsx = self.heatmap_size

        nt = self.clusters.shape[0]
        # Initiate heatmaps
        class_maps = -np.ones((vsy, vsx, nt))
        regress_maps = np.zeros((vsy, vsx, nt * 4))

        # h, w, c = img.shape
        # TODO figure out what this is supposed to do L309 in cnn_get_batch_hardmine.m
        # dx, dy = np.floor((w - self.input_size[1]) * )

        # each cluster is [-w/2, -h/2, w/2, h/2]
        dx1, dx2 = self.clusters[:, 0], self.clusters[:, 2]
        dy1, dy2 = self.clusters[:, 1], self.clusters[:, 3]

        # Filter out invalid bbox
        invalid = np.logical_or(bboxes[:, 2] <= bboxes[:, 0], bboxes[:, 3] <= bboxes[:, 1])
        ind = np.where(invalid)
        bboxes = np.delete(bboxes, ind, axis=0)

        ng = bboxes.shape[0]
        iou = np.zeros((vsy, vsx, self.clusters.shape[0], bboxes.shape[0]))

        if ng > 0:
            gx1, gy1, gx2, gy2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

            iou = compute_dense_overlap(ofx, ofy, stx, sty, vsx, vsy,
                                        dx1, dy1, dx2, dy2,
                                        gx1, gy1, gx2, gy2,
                                        1, 1)

            regress_maps, iou = self.get_regression(bboxes, [dx1, dy1, dx2, dy2], iou)

            best_iou = iou.max(axis=3)

            # Set max IoU values to 1 (even if they are < pos_thresh, as long as they are above neg_thresh)
            per_object_iou = np.reshape(iou, (-1, ng))
            fbest_idx = np.argmax(per_object_iou, axis=0)
            iou_ = np.amax(per_object_iou, axis=0)
            fbest_idx = np.unravel_index(fbest_idx[iou_ > self.neg_thresh], iou.shape[:-1])
            class_maps[fbest_idx] = 1

            # Assign positive labels
            class_maps = np.maximum(class_maps, (best_iou >= self.pos_thresh)*2-1)

            # If between positive and negative, assign as gray area
            gray = -np.ones(class_maps.shape)
            gray[np.bitwise_and(self.neg_thresh <= best_iou, best_iou < self.pos_thresh)] = 0
            class_maps = np.maximum(class_maps, gray)  # since we set the max IoU values to 1

        # handle the boundary
        non_neg_border = np.bitwise_and(pad_mask, class_maps != -1)
        class_maps[non_neg_border] = 0
        regress_maps[np.repeat(non_neg_border, 4, 2)] = 0

        # ind = np.ravel_multi_index(np.where(class_maps==1), class_maps.shape)
        # print(ind)
        # print(class_maps[np.unravel_index(ind[0], class_maps.shape)])

        # Return heatmaps
        return class_maps, regress_maps, iou

    @staticmethod
    def visualize_bboxes(image, bboxes):
        """

        :param image: PIL image
        :param bboxes:
        :return:
        """
        print("Number of GT bboxes", bboxes.shape[0])
        for idx, bbox in enumerate(bboxes):
            bbox = np.round(np.array(bbox))
            # print(bbox)
            image = draw_bounding_box(image, bbox, {"name": "car {0}".format(idx)})

        image.show(title="GT BBox")

    @staticmethod
    def render_and_save_bboxes(image, image_id, bboxes, scores, scales, directory="qualitative"):
        """
        Render the bboxes on the image and save the image
        :param image: PIL image
        :param image_id:
        :param bboxes:
        :param scores:
        :param scales:
        :param directory:
        :return:
        """
        for idx, bbox in enumerate(bboxes):
            bbox = np.round(np.array(bbox))
            image = draw_bounding_box(image, bbox, {'score': scores[idx], 'scale': scales[idx]})

        image.save("{0}/{1}.jpg".format(directory, image_id))

    def visualize_heatmaps(self, img, cls_map, reg_map, clusters, prob_thresh=1, nms_thresh=1, iou=None):
        """
        Expect cls_map and reg_map to be of the form HxWxC
        """
        fy, fx, fc = np.where(cls_map >= prob_thresh)

        # print(iou.shape)
        # best_iou = iou.max(axis=3)
        # print(best_iou.shape)
        # fy, fx, fc = np.where(best_iou >= 0.5)  # neg thresh

        cy, cx = fy*self.sty + self.ofy, fx*self.stx + self.ofx
        cw = clusters[fc, 2] - clusters[fc, 0] + 1
        ch = clusters[fc, 3] - clusters[fc, 1] + 1

        # box_ovlp = best_iou[fc, fy, fx]
        num_clusters = clusters.shape[0]

        # refine bounding box
        tx = reg_map[:, :, 0*num_clusters:1*num_clusters]
        ty = reg_map[:, :, 1*num_clusters:2*num_clusters]
        tw = reg_map[:, :, 2*num_clusters:3*num_clusters]
        th = reg_map[:, :, 3*num_clusters:4*num_clusters]

        dcx = cw * tx[fy, fx, fc]
        dcy = ch * ty[fy, fx, fc]

        rx = cx + dcx
        ry = cy + dcy

        rw = cw * np.exp(tw[fy, fx, fc])
        rh = ch * np.exp(th[fy, fx, fc])

        bboxes = np.array([np.abs(rx-rw/2), np.abs(ry-rh/2), rx+rw/2, ry+rh/2]).T

        scores = cls_map[fy, fx, fc]

        dets = np.hstack((bboxes, scores[:, np.newaxis]))
        keep = nms(dets, nms_thresh)
        bboxes = dets[keep][:, 0:4]
        # bbox_iou = best_iou[fy, fx, fc]

        # print("Best bounding box", bboxes)
        # print(bboxes.shape)

        print("Number of bboxes ", bboxes.shape[0])
        for idx, bbox in enumerate(bboxes):
            bbox = np.round(np.array(bbox))
            print(bbox)
            # img = draw_bounding_box(img, bbox, {"name": "car {0}".format(np.around(bbox_iou[idx], decimals=2))})
            img = draw_bounding_box(img, bbox, {"name": "car {0}".format(idx)})

            # if idx == 20:
            #     break

        img.show(title="Heatmap visualized")
