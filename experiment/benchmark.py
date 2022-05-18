import sys, os
from utils.tools.dataloader import BKKIterator
from utils.config import cfg
from utils.blocks.forecaster import Forecaster
from utils.blocks.encoder import Encoder
from utils.blocks.module import Predictor
from utils.tools.ordered_easydict import OrderedDict
from utils.blocks.module import EF
from torch.optim import lr_scheduler
from utils.loss import Weighted_mse_mae
from utils.blocks.trajGRU import TrajGRU
from utils.train_and_test import train_and_test
from utils.tools.evaluation import *
from net_params import *

import torch
import numpy as np
import copy
import time
import pickle


encoder = Encoder(encoder_params[0], encoder_params[1]).to(cfg.GLOBAL.DEVICE)
forecaster = Forecaster(forecaster_params[0], forecaster_params[1])

encoder_forecaster1 = EF(encoder, forecaster)
encoder_forecaster1 = encoder_forecaster1.to(cfg.GLOBAL.DEVICE)
encoder_forecaster1.load_state_dict(torch.load(os.path.join(cfg.GLOBAL.MODEL_SAVE_DIR,'trajGRU_balanced_mse_mae','models','encoder_forecaster_77000.pth')))

encoder_forecaster2 = EF(encoder, forecaster)
encoder_forecaster2 = encoder_forecaster2.to(cfg.GLOBAL.DEVICE)
encoder_forecaster2.load_state_dict(torch.load(os.path.join(cfg.GLOBAL.MODEL_SAVE_DIR,'trajGRU_frame_weighted_mse','models','encoder_forecaster_45000.pth')))

conv2d_network = Predictor(conv2d_params).to(cfg.GLOBAL.DEVICE)
conv2d_network.load_state_dict(torch.load(os.path.join(cfg.GLOBAL.MODEL_SAVE_DIR,'conv2d','models', 'encoder_forecaster_10000.pth')))


convlstm_encoder = Encoder(convlstm_encoder_params[0], convlstm_encoder_params[1]).to(cfg.GLOBAL.DEVICE)
convlstm_forecaster = Forecaster(convlstm_forecaster_params[0], convlstm_forecaster_params[1]).to(cfg.GLOBAL.DEVICE)

convlstm_encoder_forecaster = EF(convlstm_encoder, convlstm_forecaster).to(cfg.GLOBAL.DEVICE)
convlstm_encoder_forecaster.load_state_dict(torch.load(os.path.join(cfg.GLOBAL.MODEL_SAVE_DIR,'convLSTM_balacned_mse_mae','models', 'encoder_forecaster_64000.pth')))


models = OrderedDict({
    'convLSTM_balacned_mse_mae': convlstm_encoder_forecaster,
    'conv2d': conv2d_network,
    'trajGRU_balanced_mse_mae': encoder_forecaster1,
    'trajGRU_frame_weighted_mse': encoder_forecaster2,
})

model_run_avarage_time = dict()
with torch.no_grad():
    for name, model in models.items():
        is_deeplearning_model = (torch.nn.Module in model.__class__.__bases__)
        if is_deeplearning_model:
            model.eval()
        evaluator = Evaluation(seq_len=OUT_LEN, use_central=False)
        bkk_iter = BKKIterator(pd_path=cfg.ONM_PD.RAINY_TEST,
                               sample_mode="sequent",
                               seq_len=IN_LEN + OUT_LEN,
                               stride=cfg.BENCHMARK.STRIDE)
        model_run_avarage_time[name] = 0.0
        valid_time = 0
        while not bkk_iter.use_up:
            valid_batch, valid_mask, sample_datetimes, _ = bkk_iter.sample(batch_size=1)
            if valid_batch.shape[1] == 0:
                break
            if not cfg.EVALUATION.VALID_DATA_USE_UP and valid_time > cfg.EVALUATION.VALID_TIME:
                break

            valid_batch = valid_batch.astype(np.float32) / 255.0
            valid_data = valid_batch[:IN_LEN, ...]
            valid_label = valid_batch[IN_LEN:IN_LEN + OUT_LEN, ...]
            mask = valid_mask[IN_LEN:IN_LEN + OUT_LEN, ...].astype(int)

            if is_deeplearning_model:
                valid_data = torch.from_numpy(valid_data).to(cfg.GLOBAL.DEVICE)

            start = time.time()
            output = model(valid_data)
            model_run_avarage_time[name] += time.time() - start

            if is_deeplearning_model:
                output = output.cpu().numpy()

            output = np.clip(output, 0.0, 1.0)

            evaluator.update(valid_label, output, mask)

            valid_time += 1
        model_run_avarage_time[name] /= valid_time
        evaluator.save_pkl(os.path.join(cfg.BENCHMARK.STAT_PATH, name + '.pkl'))

with open(os.path.join(cfg.BENCHMARK.STAT_PATH, 'model_run_avarage_time.pkl'), 'wb') as f:
    pickle.dump(model_run_avarage_time, f)

    
for p in os.listdir(os.path.abspath(cfg.BENCHMARK.STAT_PATH)):
    e = pickle.load(open(os.path.join(cfg.BENCHMARK.STAT_PATH, p), 'rb'))
    _, _, csi, hss, _, mse, mae, balanced_mse, balanced_mae, _ = e.calculate_stat()
    print(p.split('.')[0])
    for i, thresh in enumerate(cfg.HKO.EVALUATION.THRESHOLDS):
        print('thresh %.1f csi: avarage %.4f, last frame %.4f; hss: avarage %.4f, last frame %.4f;'
              % (thresh, csi[:, i].mean(), csi[-1, i], hss[:, i].mean(), hss[-1, i]))

    print(('mse: avarage %.2f, last frame %.2f\n' +
        'mae: avarage %.2f, last frame %.2f\n'+
        'bmse: avarage %.2f, last frame %.2f\n' +
        'bmae: avarage %.2f, last frame %.2f\n') % (mse.mean(), mse[-1], mae.mean(), mae[-1],
              balanced_mse.mean(), balanced_mse[-1], balanced_mae.mean(), balanced_mae[-1]))
