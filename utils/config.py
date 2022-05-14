import os
from utils.tools.ordered_easydict import OrderedEasyDict as edict
import numpy as np
import torch

__C = edict()
cfg = __C

__C.GLOBAL = edict()
__C.GLOBAL.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
__C.GLOBAL.MAX_WORKERS = 12
__C.GLOBAL.BATCH_SIZE = 2
__C.GLOBAL.ZR = edict()
__C.GLOBAL.ZR.MAX_dBZ = 75.0
__C.GLOBAL.ZR.A = 200.0
__C.GLOBAL.ZR.B = 1.6

for dirs in ['/models/save']:
    if os.path.exists(dirs):
        __C.GLOBAL.MODEL_SAVE_DIR = dirs
assert __C.GLOBAL.MODEL_SAVE_DIR is not None

__C.ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
__C.DATA_BASE_PATH = os.path.join(__C.ROOT_DIR, 'data')

for dirs in ['/data/bkk_radar_images']:
    if os.path.exists(dirs):
        __C.RADAR_PNG_PATH = dirs
for dirs in ['/data/bkk_radar_images']:
    if os.path.exists(dirs):
        __C.PNG_PATH = dirs
for dirs in ['/data/bkk_radar_images_dBZ']:
    if os.path.exists(dirs):
        __C.MASK_PATH = dirs

__C.EVALUATION = edict()
# Image Cropping Region (TOP, LEFT, RIGHT, BOTTOM)
__C.EVALUATION.CENTRAL_REGION = (0, 0, 2034, 2048)
__C.EVALUATION.THRESHOLDS = np.array([0, 5.5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75])
__C.EVALUATION.BALANCING_WEIGHTS = (1, 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75)

__C.EVALUATION.VALID_DATA_USE_UP = True
__C.EVALUATION.VALID_TIME = 20

__C.BENCHMARK = edict()

__C.BENCHMARK.STAT_PATH = os.path.join(__C.DATA_BASE_PATH, 'benchmark_stat')
if not os.path.exists(__C.BENCHMARK.STAT_PATH):
    os.makedirs(__C.BENCHMARK.STAT_PATH)

__C.BENCHMARK.VISUALIZE_SEQ_NUM = 10  # Number of sequences that will be plotted and saved to the benchmark directory
__C.BENCHMARK.IN_LEN = 5   # The maximum input frames
__C.BENCHMARK.OUT_LEN = 20  # The maximum output frames
__C.BENCHMARK.STRIDE = 5   # The stride


# pandas data
__C.PD_BASE_PATH = os.path.join(__C.DATA_BASE_PATH, 'data/pd')

__C.IMG_DATETIME_PATH = os.path.join(__C.DATA_BASE_PATH, 'bkk_all.pkl')
# __C.SORTED_DAYS_PATH = os.path.join(__C.DATA_BASE_PATH, 'sorted_day.pkl')
# __C.RAINY_TRAIN_DAYS_PATH = os.path.join(__C.DATA_BASE_PATH, 'hko7_rainy_train_days.txt')
# __C.RAINY_VALID_DAYS_PATH = os.path.join(__C.DATA_BASE_PATH, 'hko7_rainy_valid_days.txt')
# __C.RAINY_TEST_DAYS_PATH = os.path.join(__C.DATA_BASE_PATH, 'hko7_rainy_test_days.txt')

__C.ONM_PD = edict()
# __C.ONM_PD.ALL = os.path.join(__C.ONM_PD_BASE_PATH, 'hko7_all.pkl')
# __C.ONM_PD.ALL_09_14 = os.path.join(__C.ONM_PD_BASE_PATH, 'hko7_all_09_14.pkl')
# __C.ONM_PD.ALL_15 = os.path.join(__C.ONM_PD_BASE_PATH, 'hko7_all_15.pkl')
# __C.ONM_PD.RAINY_TRAIN = os.path.join(__C.ONM_PD_BASE_PATH, 'hko7_rainy_train.pkl')
# __C.ONM_PD.RAINY_VALID = os.path.join(__C.ONM_PD_BASE_PATH, 'hko7_rainy_valid.pkl')
# __C.ONM_PD.RAINY_TEST = os.path.join(__C.ONM_PD_BASE_PATH, 'hko7_rainy_test.pkl')

__C.ONM = edict()

__C.ONM.ITERATOR = edict()
__C.ONM.ITERATOR.WIDTH = 2034
__C.ONM.ITERATOR.HEIGHT = 2048
__C.ONM.ITERATOR.FILTER_RAINFALL = True           # Whether to discard part of the rainfall, has a denoising effect
__C.ONM.ITERATOR.FILTER_RAINFALL_THRESHOLD = 0.28 # All the pixel values that are smaller than round(threshold * 255) will be discarded

__C.MODEL = edict()
from utils.blocks.activation import activation
__C.MODEL.RNN_ACT_TYPE = activation('leaky', negative_slope=0.2, inplace=True)