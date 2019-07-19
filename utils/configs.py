#coding:utf-8
# full_train.py config params

# all supported model in modelsets.py
MODEL_NAME = "simpleNet"

L2_LOSS_DECAY = 2e-4

RESTORE_MODEL = False

TRAIN_EPOCHES = 5
TRAIN_BATCHSIZE = 64
TEST_BATCHSIZE = 64

PRINT_TRAIN_INFO_PER_STEP = 20
TEST_PER_STEP = 200

INIT_LR = 1e-4
NUM_CLASSES = 10
####################################

# channels_pruning.py config params
RETRAIN_EPOCH = 3
TEST_PRUNE_BATCHES = 450
DROPOUT_TIME_THRESHOLD = 50

# pruning rate
PRUNING_RATE = 0.5
# 是否需要完成训练剪枝后的模型
FULL_TRAIN = False