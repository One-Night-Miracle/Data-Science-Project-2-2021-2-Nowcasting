from utils.tools.dataloader import BKKIterator
from utils.config import cfg
from utils.config import cfg
from utils.blocks.forecaster import Forecaster
from utils.blocks.encoder import Encoder
from collections import OrderedDict
from utils.blocks.module import EF
from torch.optim import lr_scheduler
from utils.loss import Weighted_mse_mae
from utils.blocks.trajGRU import TrajGRU
from utils.train_and_test import train_and_test
from utils.tools.evaluation import *
from utils.blocks.convLSTM import ConvLSTM
import torch
import numpy as np

batch_size = cfg.GLOBAL.BATCH_SIZE

IN_LEN = cfg.BENCHMARK.IN_LEN
OUT_LEN = cfg.BENCHMARK.OUT_LEN

# build model
encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [1, 8, 7, 5, 1]}),
        OrderedDict({'conv2_leaky_1': [64, 192, 5, 3, 1]}),
        OrderedDict({'conv3_leaky_1': [192, 192, 3, 2, 1]}),
    ],

    [
        TrajGRU(input_channel=8, num_filter=64, b_h_w=(batch_size, 96, 96), zoneout=0.0, L=13,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                act_type=cfg.MODEL.RNN_ACT_TYPE),

        TrajGRU(input_channel=192, num_filter=192, b_h_w=(batch_size, 32, 32), zoneout=0.0, L=13,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                act_type=cfg.MODEL.RNN_ACT_TYPE),
        TrajGRU(input_channel=192, num_filter=192, b_h_w=(batch_size, 16, 16), zoneout=0.0, L=9,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(3, 3), h2h_dilate=(1, 1),
                act_type=cfg.MODEL.RNN_ACT_TYPE)
    ]
]

forecaster_params = [
    [
        OrderedDict({'deconv1_leaky_1': [192, 192, 4, 2, 1]}),
        OrderedDict({'deconv2_leaky_1': [192, 64, 5, 3, 1]}),
        OrderedDict({
            'deconv3_leaky_1': [64, 8, 7, 5, 1],
            'conv3_leaky_2': [8, 8, 3, 1, 1],
            'conv3_3': [8, 1, 1, 1, 0]
        }),
    ],

    [
        TrajGRU(input_channel=192, num_filter=192, b_h_w=(batch_size, 16, 16), zoneout=0.0, L=13,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(3, 3), h2h_dilate=(1, 1),
                act_type=cfg.MODEL.RNN_ACT_TYPE),

        TrajGRU(input_channel=192, num_filter=192, b_h_w=(batch_size, 32, 32), zoneout=0.0, L=13,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                act_type=cfg.MODEL.RNN_ACT_TYPE),
        TrajGRU(input_channel=64, num_filter=64, b_h_w=(batch_size, 96, 96), zoneout=0.0, L=9,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                act_type=cfg.MODEL.RNN_ACT_TYPE)
    ]
]


# build model
conv2d_params = OrderedDict({
    'conv1_relu_1': [5, 64, 7, 5, 1],
    'conv2_relu_1': [64, 192, 5, 3, 1],
    'conv3_relu_1': [192, 192, 3, 2, 1],
    'deconv1_relu_1': [192, 192, 4, 2, 1],
    'deconv2_relu_1': [192, 64, 5, 3, 1],
    'deconv3_relu_1': [64, 64, 7, 5, 1],
    'conv3_relu_2': [64, 20, 3, 1, 1],
    'conv3_3': [20, 20, 1, 1, 0]
})


# build model
convlstm_encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [1, 8, 7, 5, 1]}),
        OrderedDict({'conv2_leaky_1': [64, 192, 5, 3, 1]}),
        OrderedDict({'conv3_leaky_1': [192, 192, 3, 2, 1]}),
    ],

    [
        ConvLSTM(input_channel=8, num_filter=64, b_h_w=(batch_size, 96, 96),
                 kernel_size=3, stride=1, padding=1),
        ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 32, 32),
                 kernel_size=3, stride=1, padding=1),
        ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 16, 16),
                 kernel_size=3, stride=1, padding=1),
    ]
]

convlstm_forecaster_params = [
    [
        OrderedDict({'deconv1_leaky_1': [192, 192, 4, 2, 1]}),
        OrderedDict({'deconv2_leaky_1': [192, 64, 5, 3, 1]}),
        OrderedDict({
            'deconv3_leaky_1': [64, 8, 7, 5, 1],
            'conv3_leaky_2': [8, 8, 3, 1, 1],
            'conv3_3': [8, 1, 1, 1, 0]
        }),
    ],

    [
        ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 16, 16),
                 kernel_size=3, stride=1, padding=1),
        ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 32, 32),
                 kernel_size=3, stride=1, padding=1),
        ConvLSTM(input_channel=64, num_filter=64, b_h_w=(batch_size, 96, 96),
                 kernel_size=3, stride=1, padding=1),
    ]
]
