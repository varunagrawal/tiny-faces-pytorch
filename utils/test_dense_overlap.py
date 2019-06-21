from .dense_overlap import compute_dense_overlap
from scipy.io import loadmat
import numpy as np


d = loadmat("dense_overlap.mat")

ofx, ofy = d['ofx'][0, 0], d['ofy'][0, 0]
stx, sty = d['stx'][0, 0], d['sty'][0, 0]
vsx, vsy = d['vsx'][0, 0], d['vsy'][0, 0]
dx1, dy1, dx2, dy2 = d['dx1'], d['dy1'], d['dx2'], d['dy2']
dx1 = dx1.reshape(dx1.shape[2])
dy1 = dy1.reshape(dy1.shape[2])
dx2 = dx2.reshape(dx2.shape[2])
dy2 = dy2.reshape(dy2.shape[2])

gx1, gy1, gx2, gy2 = d['gx1'], d['gy1'], d['gx2'], d['gy2']
gx1 = gx1.reshape(gx1.shape[0])
gy1 = gy1.reshape(gy1.shape[0])
gx2 = gx2.reshape(gx2.shape[0])
gy2 = gy2.reshape(gy2.shape[0])

correct_iou = d['iou']

iou = compute_dense_overlap(ofx, ofy, stx, sty, vsx, vsy,
                            dx1, dy1, dx2, dy2,
                            gx1, gy1, gx2, gy2,
                            1, 1)

print("Computed IOU")
print("iou shape", iou.shape)
print("correct iou shape", correct_iou.shape)
print("Tensors are close enough?", np.allclose(iou, correct_iou))
print("Tensors are equal?", np.array_equal(iou, correct_iou))
