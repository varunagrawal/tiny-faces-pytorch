.RECIPEPREFIX +=

PYTHON=python3
ROOT=data/WIDER
TRAINDATA=$(ROOT)/wider_face_split/wider_face_train_bbx_gt.txt
VALDATA=$(ROOT)/wider_face_split/wider_face_val_bbx_gt.txt
TESTDATA=$(ROOT)/wider_face_split/wider_face_test_filelist.txt

CHECKPOINT=weights/checkpoint_50.pth

main: 
        $(PYTHON) main.py $(TRAINDATA) $(VALDATA) --dataset-root $(ROOT)

resume: 
        $(PYTHON) main.py $(TRAINDATA) $(VALDATA) --dataset-root $(ROOT) --resume $(CHECKPOINT) --epochs $(EPOCH)

evaluate: 
        $(PYTHON) evaluate.py $(VALDATA) --dataset-root $(ROOT) --checkpoint $(CHECKPOINT) --split val

evaluation:
        cd eval_tools/ && octave wider_eval.m

test: 
        $(PYTHON) evaluate.py $(TESTDATA) --dataset-root $(ROOT) --checkpoint $(CHECKPOINT) --split test

cluster: 
        cd utils; $(PYTHON) cluster.py $(TRAIN_INSTANCES)

debug: 
        $(PYTHON) main.py $(TRAINDATA) $(VALDATA) --dataset-root $(ROOT) --batch_size 1 --workers 0 --debug

debug-evaluate: 
        $(PYTHON) evaluate.py $(VALDATA) --dataset-root $(ROOT) --checkpoint $(CHECKPOINT) --split val --batch_size 1 --workers 0 --debug
