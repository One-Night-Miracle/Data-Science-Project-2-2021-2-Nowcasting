import pandas as pd
from utils.config import cfg
from utils.utils import *

rebuild_bkk_pkl()

def train_test_split(pd_path, n_data=500, ratio=(0.8,0.5,0.15)):
    assert n_data > 500
    assert ratio[0] + ratio[1] + ratio[2] == 1 
    bkk_data = pd.read_pickle(cfg.ONM_PD.FOLDER_ALL)[:n_data]

    ratio[1] = ratio[0] + ratio[1]
    ratio[2] = ratio[1] + ratio[2]

    n_train = ratio[0]*bkk_data.shape[0]//1
    n_valid = ratio[1]*bkk_data.shape[0]//1
    n_test = ratio[2]*bkk_data.shape[0]//1

    bkk_train = bkk_data.iloc[:n_train]
    bkk_valid = bkk_data.iloc[n_train:n_valid]
    bkk_test = bkk_data.iloc[n_valid:]

    bkk_train.to_csv(cfg.ONM_CSV.RAINY_TRAIN)
    pd.to_pickle(bkk_train, cfg.ONM_PD.RAINY_TRAIN)

    bkk_valid.to_csv(cfg.ONM_CSV.RAINY_VALID)
    pd.to_pickle(bkk_valid, cfg.ONM_PD.RAINY_VALID)

    bkk_test.to_csv(cfg.ONM_CSV.RAINY_TEST)
    pd.to_pickle(bkk_test, cfg.ONM_PD.RAINY_TEST)

