import torch
from torch import nn


class AvgMeter:
    def __init__(self):
        self.average = 0
        self.num_averaged = 0

    def update(self, loss, sz):
        n = self.num_averaged
        m = n + sz
        self.average = ((n * self.average) + float(loss)) / m
        self.num_averaged = m

    def reset(self):
        self.average = 0
        self.num_averaged = 0


class DetectionCriterion(nn.Module):
    """
    The loss for the Tiny Faces detector
    """
    def __init__(self, n_templates=25, reg_weight=2):
        super().__init__()

        # We don't want per element averaging.
        # We want to normalize over the batch or positive samples.
        self.regression_criterion = nn.SmoothL1Loss(reduction='none')
        self.classification_criterion = nn.SoftMarginLoss(reduction='none')
        self.n_templates = n_templates
        self.reg_weight = reg_weight

        self.cls_average = AvgMeter()
        self.reg_average = AvgMeter()

        self.cls_mask = None
        self.masked_cls_loss = None
        self.masked_reg_loss = None
        self.reg_mask = None
        self.total_loss = None

    def hard_negative_mining(self, classification, class_map):
        cls_loss = nn.functional.soft_margin_loss(classification, class_map, reduction='none')
        class_map[cls_loss < 0.03] = 0
        return class_map

    def forward(self, output, class_map, regression_map):
        classification = output[:, 0:self.n_templates, :, :]
        regression = output[:, self.n_templates:, :, :]

        # hard negative mining
        class_map = self.hard_negative_mining(classification, class_map)

        # weights used to mask out invalid regions i.e. where the label is 0
        self.cls_mask = (class_map != 0).type(output.dtype)

        cls_loss = self.classification_criterion(classification, class_map)

        # Mask the classification loss
        self.masked_cls_loss = self.cls_mask * cls_loss

        reg_loss = self.regression_criterion(regression, regression_map)
        # make same size as reg_map
        self.reg_mask = (class_map > 0).repeat(1, 4, 1, 1).type(output.dtype)

        self.masked_reg_loss = self.reg_mask * reg_loss  # / reg_loss.size(0)

        self.total_loss = self.masked_cls_loss.sum() + self.reg_weight*self.masked_reg_loss.sum()

        self.cls_average.update(self.masked_cls_loss.sum(), output.size(0))
        self.reg_average.update(self.masked_reg_loss.sum(), output.size(0))

        return self.total_loss

    def reset(self):
        self.cls_average.reset()
        self.reg_average.reset()
