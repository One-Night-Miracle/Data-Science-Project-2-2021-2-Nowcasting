import pandas as pd
from utils.config import cfg
from utils.utils import *

def train_test_split(pd_path, n=-1, ratio=(0.8,0.05,0.15)):
    assert ratio[0] + ratio[1] + ratio[2] == 1 
    rebuild_bkk_pkl()
    bkk_data = pd.read_pickle(cfg.ONM_PD.FOLDER_ALL)[:n]

    ratio_train = int(ratio[0]*bkk_data.shape[0])
    ratio_valid = int((ratio[0] + ratio[1])*bkk_data.shape[0])

    bkk_train = bkk_data.iloc[:ratio_train]
    bkk_valid = bkk_data.iloc[ratio_train:ratio_valid]
    bkk_test = bkk_data.iloc[ratio_valid:]

    bkk_train.to_csv(cfg.ONM_CSV.RAINY_TRAIN)
    pd.to_pickle(bkk_train, cfg.ONM_PD.RAINY_TRAIN)

    bkk_valid.to_csv(cfg.ONM_CSV.RAINY_VALID)
    pd.to_pickle(bkk_valid, cfg.ONM_PD.RAINY_VALID)

    bkk_test.to_csv(cfg.ONM_CSV.RAINY_TEST)
    pd.to_pickle(bkk_test, cfg.ONM_PD.RAINY_TEST)

