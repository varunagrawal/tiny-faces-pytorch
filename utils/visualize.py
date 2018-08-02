from PIL import Image, ImageDraw, ImageFont
import json
from pathlib import Path
import numpy as np


def draw_bounding_box(img, bbox, labels):
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    color = tuple(np.random.choice(range(100, 256), size=3))

    draw.rectangle((bbox[0], bbox[1], bbox[2], bbox[3]), outline=color)

    for i, k in enumerate(labels.keys()):
        w, h = font.getsize(labels[k])
        # draw.rectangle((bbox[0], bbox[1] + i*h, bbox[0] + w, bbox[1] + (i+2)*h), fill=color)
        draw.text((bbox[0], bbox[1] + i*h), "{0}:{1:.3f} ".format(k, labels[k]), fill=color)

    return img


def draw_all_boxes(img, bboxes, categories):
    for bbox, c in zip(bboxes, categories):
        img = draw_bounding_box(img, bbox, c)

    img.show()


def visualize_predictions(split="train", year="2014"):
    predictions = open("../predictions.ndjson").readlines()
    root = "/home/vagrawal/datasets/coco/{year}/{split}{year}".format(split=split, year=year)

    for pred in predictions:
        d = json.loads(pred)
        image_path = Path(root, "COCO_train{year}_{id:012}.jpg".format(year=year, id=d["id"]))
        img = Image.open(image_path)
        bboxes = d["bboxes"]

        for bbox in bboxes:
            bbox = np.round(np.array(bbox))
            img = draw_bounding_box(img, bbox, {"name": "car"})

        img.show()


if __name__ == "__main__":
    visualize_predictions()
