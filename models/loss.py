import numpy as np
import torch
from torch import nn


class AvgMeter:
    def __init__(self):
        self.average = 0
        self.num_averaged = 0

    def update(self, loss, size):
        n = self.num_averaged
        m = n + size
        self.average = ((n * self.average) + float(loss)) / m
        self.num_averaged = m

    def reset(self):
        self.average = 0
        self.num_averaged = 0


def balance_sampling(label_cls, pos_fraction, sample_size=256):
    """
    Perform balance sampling by always sampling `pos_fraction` positive samples and
    `(1-pos_fraction)` negative samples from the input
    :param label_cls: Class labels as numpy.array.
    :param pos_fraction: The maximum fraction of positive samples to keep.
    :return:
    """
    pos_maxnum = sample_size * pos_fraction  # sample 128 positive points

    # Find all the points where we have objects and ravel the indices to get a 1D array.
    # This makes the subsequent operations easier to reason about
    pos_idx_unraveled = np.where(label_cls == 1)
    pos_idx = np.array(np.ravel_multi_index(
        pos_idx_unraveled, label_cls.shape))

    if pos_idx.size > pos_maxnum:
        # Get all the indices of the locations to be zeroed out
        didx = shuffle_index(pos_idx.size, pos_idx.size-pos_maxnum)
        # Get the locations and unravel it so we can index
        pos_idx_unraveled = np.unravel_index(pos_idx[didx], label_cls.shape)
        label_cls[pos_idx_unraveled] = 0

    neg_maxnum = pos_maxnum * (1 - pos_fraction) / pos_fraction
    neg_idx_unraveled = np.where(label_cls == -1)
    neg_idx = np.array(np.ravel_multi_index(neg_idx_unraveled,
                                            label_cls.shape))

    if neg_idx.size > neg_maxnum:
        # Get all the indices of the locations to be zeroed out
        ridx = shuffle_index(neg_idx.size, neg_maxnum)
        didx = np.arange(0, neg_idx.size)
        didx = np.delete(didx, ridx)
        neg_idx = np.unravel_index(neg_idx[didx], label_cls.shape)
        label_cls[neg_idx] = 0

    return label_cls


def shuffle_index(n, n_out):
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


class DetectionCriterion(nn.Module):
    """
    The loss for the Tiny Faces detector
    """

    def __init__(self, n_templates=25, reg_weight=2, pos_fraction=0.5):
        super().__init__()

        # We don't want per element averaging.
        # We want to normalize over the batch or positive samples.
        self.regression_criterion = nn.SmoothL1Loss(reduction='none')
        self.classification_criterion = nn.SoftMarginLoss(reduction='none')
        self.n_templates = n_templates
        self.reg_weight = reg_weight
        self.pos_fraction = pos_fraction

        self.cls_average = AvgMeter()
        self.reg_average = AvgMeter()

        self.cls_mask = None
        self.masked_cls_loss = None
        self.masked_reg_loss = None
        self.reg_mask = None
        self.total_loss = None

    def balance_sample(self, class_map):
        device = class_map.device
        label_cls_np = class_map.cpu().numpy()
        # iterate through batch
        for idx in range(label_cls_np.shape[0]):
            label_cls_np[idx, ...] = balance_sampling(label_cls_np[idx, ...],
                                                      pos_fraction=self.pos_fraction)

        class_map = torch.from_numpy(label_cls_np).to(device)

        return class_map

    def hard_negative_mining(self, classification, class_map):
        loss_class_map = nn.functional.soft_margin_loss(classification.detach(), class_map,
                                                        reduction='none')
        class_map[loss_class_map < 0.03] = 0
        return class_map

    def forward(self, output, class_map, regression_map):
        classification = output[:, 0:self.n_templates, :, :]
        regression = output[:, self.n_templates:, :, :]

        # online hard negative mining
        class_map = self.hard_negative_mining(classification, class_map)
        # balance sampling
        class_map = self.balance_sample(class_map)

        # weights used to mask out invalid regions i.e. where the label is 0
        self.cls_mask = (class_map != 0).type(output.dtype)

        cls_loss = self.classification_criterion(classification, class_map)

        # Mask the classification loss
        self.masked_cls_loss = self.cls_mask * cls_loss

        reg_loss = self.regression_criterion(regression, regression_map)
        # make same size as reg_map
        self.reg_mask = (class_map > 0).repeat(1, 4, 1, 1).type(output.dtype)

        self.masked_reg_loss = self.reg_mask * reg_loss  # / reg_loss.size(0)

        self.total_loss = self.masked_cls_loss.sum() + \
            self.reg_weight * self.masked_reg_loss.sum()

        self.cls_average.update(self.masked_cls_loss.sum(), output.size(0))
        self.reg_average.update(self.masked_reg_loss.sum(), output.size(0))

        return self.total_loss

    def reset(self):
        self.cls_average.reset()
        self.reg_average.reset()
