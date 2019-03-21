.RECIPEPREFIX +=

PYTHON=python3
ROOT=data/WIDER
TRAINDATA=$(ROOT)/wider_face_split/wider_face_train_bbx_gt.txt
VALDATA=$(ROOT)/wider_face_split/wider_face_val_bbx_gt.txt

CHECKPOINT=weights/checkpoint_50.pth

main: cython
        $(PYTHON) main.py $(TRAINDATA) --dataset-root $(ROOT)

resume: cython
        $(PYTHON) main.py $(TRAINDATA) --dataset-root $(ROOT) --resume $(CHECKPOINT) --epochs $(EPOCH)

evaluate: cython
        $(PYTHON) evaluate.py $(VALDATA) --dataset-root $(ROOT) --checkpoint $(CHECKPOINT)

cluster: cython
        cd utils; $(PYTHON) cluster.py $(TRAIN_INSTANCES)

debug: cython
        $(PYTHON) main.py $(TRAINDATA) --dataset-root $(ROOT) --batch-size 1 --workers 0

cython:
        $(PYTHON) setup.py build_ext --inplace
