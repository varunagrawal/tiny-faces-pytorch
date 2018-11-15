import json
import os.path as osp
from pathlib import Path
from PIL import Image
from skimage import transform, img_as_ubyte
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import dataset
from .processor import DataProcessor
import warnings


class COCO(dataset.Dataset):
    def __init__(self, path, clusters, img_transforms=None, dataset_root="", train=True, split="train",
                 input_size=(512, 512), heatmap_size=(64, 64), multiscale=False,
                 pos_thresh=0.7, neg_thresh=0.3, pos_fraction=0.5,
                 model_downsample_factor=8, debug=False):
        """
        Each bounding box in COCO is [x, y, w, h]. We convert it to [x1, y1, x2, y2] when we `get` it.
        :param path:
        :param clusters:
        :param img_transforms:
        :param dataset_root:
        :param train:
        :param input_size:
        :param heatmap_size:
        :param pos_thresh:
        :param neg_thresh:
        """
        super().__init__()

        print("Loading dataset")
        data = json.load(open(path))

        # self.data = data
        self.data = []
        for x in data:
            categories = [b['category_id'] for b in x['bboxes']]
            if 3 in categories:  # if there is a car bounding box
                # filter out the bounding boxes for only cars
                x['bboxes'] = [b for b in x['bboxes'] if b['category_id'] == 3]
                # only add images which have cars in them
                self.data.append(x)

        print("Dataset loaded")
        print("{0} samples in the dataset".format(len(self.data)))

        self.clusters = clusters
        self.transforms = img_transforms
        self.dataset_root = Path(osp.expanduser(dataset_root))
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.pos_thresh = pos_thresh
        self.neg_thresh = neg_thresh
        self.pos_fraction = pos_fraction
        self.multiscale = multiscale
        self.model_downsample_factor = model_downsample_factor

        # receptive field computed using a combination of values from Matconvnet plus derived equations.
        self.rf = {
            'size': [859, 859],
            'stride': [model_downsample_factor, model_downsample_factor],
            'offset': [-2, -2]  # computed as per 0 indexing in Python
        }

        self.train = train
        self.split = split
        self.year = 2014

        self.processor = DataProcessor(input_size, heatmap_size, pos_thresh, neg_thresh, clusters, rf=self.rf)
        self.debug = debug

    def get_all_bboxes(self):
        bboxes = np.empty((0, 4))
        for d in self.data:
            for b in d['bboxes']:
                bboxes = np.vstack((bboxes, b['bbox']))

        # Each bbox is (x, y, w, h) so we convert to (x1, y1, x2, y2)
        bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2] - 1
        bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3] - 1

        return bboxes

    def process_inputs(self, image, bboxes):
        """
        ## RF size = [859, 859], stride = [8, 8], offset = [-2, -2]
        :param image:
        :param bboxes:
        :return:
        """
        img = np.array(image)

        # Randomly resize the image
        rnd = np.random.rand()
        if rnd < 1 / 3:
            # resize by half
            img = transform.rescale(img, 0.5, multichannel=True, mode='reflect', anti_aliasing=True)
            # scaled_shape = (int(0.5 * image.height), int(0.5 * image.width))
            # image = transforms.functional.resize(image, scaled_shape)

            bboxes = bboxes / 2
        elif rnd > 2 / 3:
            # double size
            img = transform.rescale(img, 2, multichannel=True, mode='reflect', anti_aliasing=True)
            # scaled_shape = (int(2 * image.height), int(2 * image.width))
            # image = transforms.functional.resize(image, scaled_shape)
            bboxes = bboxes * 2

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # skimage scales from 255 to -1/1 range, thus we scale it back
            img = img_as_ubyte(img)
        # img = np.array(image)

        img, bboxes, paste_box = self.processor.crop_image(img, bboxes)
        pad_mask = self.processor.get_padding(paste_box)

        # Random Flip
        if np.random.rand() > 0.5:
            img = np.fliplr(img).copy()  # flip the image
            pad_mask = np.fliplr(pad_mask).copy()  # flip the padding mask
            lx1, lx2 = np.array(bboxes[:, 0]), np.array(bboxes[:, 2])
            bboxes[:, 0] = self.input_size[1] - lx2 - 1
            bboxes[:, 2] = self.input_size[1] - lx1 - 1  # Flip the bounding box. -1 for correct indexing

        class_maps, regress_maps, iou = self.processor.get_heatmaps(bboxes, pad_mask)

        # perform balance sampling so there are roughly the same number of positive and negative samples.
        class_maps = self.processor.balance_sampling(class_maps, self.pos_fraction)

        if self.debug:
            # Confirm is balance sampling works
            print(class_maps[class_maps == 1].sum())
            print(class_maps[class_maps == -1].sum())

            # Visualize stuff
            self.processor.visualize_bboxes(Image.fromarray(img.astype('uint8'), 'RGB'), bboxes)
            self.processor.visualize_heatmaps(Image.fromarray(img.astype('uint8'), 'RGB'),
                                              class_maps, regress_maps, self.clusters, iou=iou)
            # and now we exit
            exit(0)

        # transpose so we get CxHxW
        class_maps = class_maps.transpose((2, 0, 1))
        regress_maps = regress_maps.transpose((2, 0, 1))

        # img is type float64. Convert it to uint8 so torch knows to treat it like an image
        img = img.astype(np.uint8)

        return img, class_maps, regress_maps

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        ann = self.data[index]
        image_id = ann['image']["id"]

        if self.debug:
            print(index, image_id)

        ground_truth = ann['bboxes']

        labels = [x['category_id'] for x in ground_truth]

        image_file = ann['image']['file_name']

        image = Image.open(self.dataset_root /
                           "{0}{1}".format(self.split, self.year) /
                           image_file).convert('RGB')

        # filter the annotations, so we only have cars
        filtered_anns = [x for i, x in enumerate(ground_truth) if labels[i] == 3]

        bboxes = np.zeros((len(filtered_anns), 4))
        for i, b in enumerate(filtered_anns):
            bboxes[i, :] = np.array(b['bbox'])

        # convert to (x1, y1, x2, y2)
        # We work with the two point representation since cropping and resizing becomes easier to deal with
        bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2] - 1
        bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3] - 1

        labels = np.array(labels, ndmin=2)  # set the leading dimension to 1 so that we can batch

        if self.split == 'train':
            img, class_map, reg_map = self.process_inputs(image, bboxes)

            # convert everything to tensors
            if self.transforms is not None:
                # if img is a byte or uint8 array, it will convert from 0-255 to 0-1
                # this converts from (HxWxC) to (CxHxW) as well
                img = self.transforms(img)

            class_map, reg_map = torch.from_numpy(class_map), torch.from_numpy(reg_map)

            return img, class_map, reg_map

        elif self.split == 'val':
            # needed because a dataloader doesn't accept PIL images for batching
            img = transforms.ToTensor()(image)
            return img, image_id, labels