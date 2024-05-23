import numpy as np
from scipy.io import loadmat

from tinyfaces.utils.metrics import rect_dist


def test_rect_dist(x, y, gt_dist):
    d = rect_dist(x, y)
    print("Is my rect_dist code correct?", np.array_equal(d, gt_dist))


def main():
    truth = loadmat('rect_dist.mat')
    gt_dist = truth['d'][:, 0]
    x = truth['labelRect']
    y = truth['tLabelRect']
    test_rect_dist(x, y, gt_dist)
    print(rect_dist(x[0, :], y[0, :]))


if __name__ == "__main__":
    main()
