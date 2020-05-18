from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import dataset
from torchvision import transforms

from utils import visualize

from .processor import DataProcessor


class WIDERFace(dataset.Dataset):
    """The WIDERFace dataset is generated using MATLAB,
    so a lot of small housekeeping elements have been added
    to take care of the indexing discrepancies."""

    def __init__(self, path, templates, img_transforms=None, dataset_root="", split="train",
                 train=True, input_size=(500, 500), heatmap_size=(63, 63),
                 pos_thresh=0.7, neg_thresh=0.3, pos_fraction=0.5, debug=False):
        super().__init__()

        self.data = []
        self.split = split

        self.load(path)

        print("Dataset loaded")
        print("{0} samples in the {1} dataset".format(len(self.data),
                                                      self.split))
        # self.data = data

        # canonical object templates obtained via clustering
        # NOTE we directly use the values from Peiyun's repository stored in "templates.json"
        self.templates = templates

        self.transforms = img_transforms
        self.dataset_root = Path(dataset_root)
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.pos_thresh = pos_thresh
        self.neg_thresh = neg_thresh
        self.pos_fraction = pos_fraction

        # receptive field computed using a combination of values from Matconvnet
        # plus derived equations.
        self.rf = {
            'size': [859, 859],
            'stride': [8, 8],
            'offset': [-1, -1]
        }

        self.processor = DataProcessor(input_size, heatmap_size,
                                       pos_thresh, neg_thresh,
                                       templates, rf=self.rf)
        self.debug = debug

    def load(self, path):
        """Load the dataset from the text file."""

        if self.split in ("train", "val"):
            lines = open(path).readlines()
            self.data = []
            idx = 0

            while idx < len(lines):
                img = lines[idx].strip()
                idx += 1
                n = int(lines[idx].strip())
                idx += 1

                bboxes = np.empty((n, 10))

                if n == 0:
                    idx += 1
                else:
                    for b in range(n):
                        bboxes[b, :] = [abs(float(x))
                                        for x in lines[idx].strip().split()]
                        idx += 1

                # remove invalid bboxes where w or h are 0
                invalid = np.where(np.logical_or(bboxes[:, 2] == 0,
                                                 bboxes[:, 3] == 0))
                bboxes = np.delete(bboxes, invalid, 0)

                # bounding boxes are 1 indexed so we keep them like that
                # and treat them as abstract geometrical objects
                # We only need to worry about the box indexing when actually rendering them

                # convert from (x, y, w, h) to (x1, y1, x2, y2)
                # We work with the two point representation
                # since cropping becomes easier to deal with
                # -1 to ensure the same representation as in Matlab.
                bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2] - 1
                bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3] - 1

                datum = {
                    "img_path": img,
                    "bboxes": bboxes[:, 0:4],
                    "blur": bboxes[:, 4],
                    "expression": bboxes[:, 5],
                    "illumination": bboxes[:, 6],
                    "invalid": bboxes[:, 7],
                    "occlusion": bboxes[:, 8],
                    "pose": bboxes[:, 9]
                }

                self.data.append(datum)

        elif self.split == "test":
            data = open(path).readlines()
            self.data = [{'img_path': x.strip()} for x in data]

    def get_all_bboxes(self):
        bboxes = np.empty((0, 4))
        for datum in self.data:
            bboxes = np.vstack((bboxes, datum['bboxes']))

        return bboxes

    def __len__(self):
        return len(self.data)

    def process_inputs(self, image, bboxes):
        # Randomly resize the image
        rnd = np.random.rand()
        if rnd < 1 / 3:
            # resize by half
            scaled_shape = (int(0.5 * image.height), int(0.5 * image.width))
            image = transforms.functional.resize(image, scaled_shape)
            bboxes = bboxes / 2

        elif rnd > 2 / 3:
            # double size
            scaled_shape = (int(2 * image.height), int(2 * image.width))
            image = transforms.functional.resize(image, scaled_shape)
            bboxes = bboxes * 2

        # convert from PIL Image to ndarray
        img = np.array(image)

        # Get a random crop of the image and keep only relevant bboxes
        img, bboxes, paste_box = self.processor.crop_image(img, bboxes)
        pad_mask = self.processor.get_padding(paste_box)

        # Random Flip
        flip = np.random.rand() > 0.5
        if flip:
            img = np.fliplr(img).copy()  # flip the image

            lx1, lx2 = np.array(bboxes[:, 0]), np.array(bboxes[:, 2])
            # Flip the bounding box. +1 for correct indexing
            bboxes[:, 0] = self.input_size[1] - lx2 + 1
            bboxes[:, 2] = self.input_size[1] - lx1 + 1

            pad_mask = np.fliplr(pad_mask)

        # Get the ground truth class and regression maps
        class_maps, regress_maps, iou = self.processor.get_heatmaps(bboxes,
                                                                    pad_mask)

        if self.debug:
            # Visualize stuff
            visualize.visualize_bboxes(Image.fromarray(img.astype('uint8'), 'RGB'),
                                       bboxes)
            self.processor.visualize_heatmaps(Image.fromarray(img.astype('uint8'), 'RGB'),
                                              class_maps, regress_maps, self.templates, iou=iou)

            # and now we exit
            exit(0)

        # transpose so we get CxHxW
        class_maps = class_maps.transpose((2, 0, 1))
        regress_maps = regress_maps.transpose((2, 0, 1))

        # img is type float64. Convert it to uint8 so torch knows to treat it like an image
        img = img.astype(np.uint8)

        return img, class_maps, regress_maps, bboxes

    def __getitem__(self, index):
        datum = self.data[index]

        image_root = self.dataset_root / "WIDER_{0}".format(self.split)
        image_path = image_root / "images" / datum['img_path']
        image = Image.open(image_path).convert('RGB')

        if self.split == 'train':
            bboxes = datum['bboxes']

            if self.debug:
                if bboxes.shape[0] == 0:
                    print(image_path)
                print("Dataset index: \t", index)
                print("image path:\t", image_path)

            img, class_map, reg_map, bboxes = self.process_inputs(image,
                                                                  bboxes)

            # convert everything to tensors
            if self.transforms is not None:
                # if img is a byte or uint8 array, it will convert from 0-255 to 0-1
                # this converts from (HxWxC) to (CxHxW) as well
                img = self.transforms(img)

            class_map = torch.from_numpy(class_map)
            reg_map = torch.from_numpy(reg_map)

            return img, class_map, reg_map

        elif self.split == 'val':
            # NOTE Return only the image and the image path.
            # Use the eval_tools to get the final results.
            if self.transforms is not None:
                # Only convert to tensor since we do normalization after rescaling
                img = transforms.functional.to_tensor(image)

            return img, datum['img_path']

        elif self.split == 'test':
            filename = datum['img_path']

            if self.transforms is not None:
                img = self.transforms(image)

            return img, filename
