.RECIPEPREFIX +=

PYTHON=python3
ROOT=data/WIDER
TRAINDATA=$(ROOT)/wider_face_split/wider_face_train_bbx_gt.txt
VALDATA=$(ROOT)/wider_face_split/wider_face_val_bbx_gt.txt
TESTDATA=$(ROOT)/wider_face_split/wider_face_test_filelist.txt

CHECKPOINT=weights/checkpoint_50.pth

main: cython
        $(PYTHON) main.py $(TRAINDATA) $(VALDATA) --dataset-root $(ROOT)

resume: cython
        $(PYTHON) main.py $(TRAINDATA) $(VALDATA) --dataset-root $(ROOT) --resume $(CHECKPOINT) --epochs $(EPOCH)

evaluate: cython
        $(PYTHON) evaluate.py $(VALDATA) --dataset-root $(ROOT) --checkpoint $(CHECKPOINT) --split val

test: cython
        $(PYTHON) evaluate.py $(TESTDATA) --dataset-root $(ROOT) --checkpoint $(CHECKPOINT) --split test

cluster: cython
        cd utils; $(PYTHON) cluster.py $(TRAIN_INSTANCES)

debug: cython
        $(PYTHON) main.py $(TRAINDATA) --dataset-root $(ROOT) --batch-size 1 --workers 0

cython:
        $(PYTHON) setup.py build_ext --inplace
