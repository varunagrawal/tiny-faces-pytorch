"""
Function to process the dataset and other files into a format that maps each image to all of its bounding boxes.
"""
import json
from pathlib import Path
from tqdm import tqdm
import argparse


def process_bbox(dataset):
    bboxes = {}
    images = {}

    images_list = dataset['images']
    annotations = dataset["annotations"]

    for image in tqdm(images_list, total=len(images_list)):
        images[image['id']] = image

    for ann in tqdm(annotations, total=len(annotations)):
        if ann["image_id"] in bboxes.keys():
            bboxes[ann["image_id"]]['bboxes'].append(ann)
        else:
            bboxes[ann['image_id']] = {
                'image': images[ann['image_id']],
                'bboxes': [ann]
            }

    data = [bboxes[key] for key in bboxes.keys()]
    return data


parser = argparse.ArgumentParser()
parser.add_argument('root', default="~/datasets/coco/annotations")
parser.add_argument('--year', default="2014", help="Dataset release year. Default: 2014")
args = parser.parse_args()

splits = [('instances_train{0}.json'.format(args.year), 'train'), ('instances_val{0}.json'.format(args.year), 'val')]

for dataset, split in splits:
    print(split)
    data = json.load(open(Path(args.root, dataset)))
    data = process_bbox(data)
    json.dump(data, open(Path(args.root, "bounding_box_annotations_{0}.json".format(split)), 'w'))
