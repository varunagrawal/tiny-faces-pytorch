# tiny-faces-pytorch

This is a PyTorch implementation of Peiyun Hu's [awesome tiny face detector](https://github.com/peiyunh/tiny). 

We use (and recommend) **Python 3.6+** for minimal pain when using this codebase (plus Python 3.6 has really cool features).

**NOTE** Be sure to cite Peiyun's CVPR paper if you use this repository!

## Getting Started

We assume Python 
- Clone this repository.
- Download the WIDER Face dataset and annotations files to `data/WIDER`.
- Install dependencies with `pip instll -r requirements.txt`.

Your data directory should look like this for WIDERFace

```
- data
    - WIDER
        - README.md
        - wider_face_split
        - WIDER_train
        - WIDER_val
        - WIDER_test
```

## Training

Just type `make` at the repo root and you should be good to go!

In case you wish to change some settings (such as data location), you can modify the `Makefile` which should be super easy to work with.

## Evaluation

WIP
