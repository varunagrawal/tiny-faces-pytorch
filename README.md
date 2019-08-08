# tiny-faces-pytorch

This is a PyTorch implementation of Peiyun Hu's [awesome tiny face detector](https://github.com/peiyunh/tiny). 

We use (and recommend) **Python 3.6+** for minimal pain when using this codebase (plus Python 3.6 has really cool features).

**NOTE** Be sure to cite Peiyun's CVPR paper and this repo if you use this code!

This code gives the following mAP results on the WIDER Face dataset:

| Setting | mAP   |
|---------|-------|
| easy    | 0.902 |
| medium  | 0.892 |
| hard    | 0.797 |

## Getting Started

- Clone this repository.
- Download the WIDER Face dataset and annotations files to `data/WIDER`.
- Install dependencies with `pip install -r requirements.txt`.

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

## Pretrained Weights

You can find the pretrained weights which get the above mAP results [here](https://drive.google.com/open?id=1V8c8xkMrQaCnd3MVChvJ2Ge-DUfXPHNu).

## Training

Just type `make` at the repo root and you should be good to go!

In case you wish to change some settings (such as data location), you can modify the `Makefile` which should be super easy to work with.

## Evaluation

To run evaluation and generate the output files as per the WIDERFace specification, simply run `make evaluate`. The results will be stored in the `val_results` directory.

You can then use the dataset's `eval_tools` to generate the mAP numbers (this needs Matlab/Octave).

Similarly, to run the model on the test set, run `make test` to generate results in the `test_results` directory.
