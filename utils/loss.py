from torch import nn
import torch
from utils.config import cfg
from utils.utils import dBZ_to_pixel
import torch.nn.functional as F

class WeightedCrossEntropyLoss(nn.Module):

    # Weight should be a 1D Tensor assigning weight to each of the classes.
    def __init__(self, weight=None, LAMBDA=None):
        super().__init__()
        self._weight = weight
        self._lambda = LAMBDA

    def forward(self, input, target, mask):
        assert input.size(0) == cfg.BENCHMARK.OUT_LEN

        # B*C*S*H*W
        input = input.permute((1, 2, 0, 3, 4))

        # B*S*H*W
        target = target.permute((1, 2, 0, 3, 4)).squeeze(1)

        class_index = torch.zeros_like(target).long()
        thresholds = [0.0] + [dBZ_to_pixel(ele) for ele in cfg.EVALUATION.THRESHOLDS]

        for i, threshold in enumerate(thresholds):
            class_index[target >= threshold] = i
        
        # F.cross_entropy should be B*C*S*H*W
        error = F.cross_entropy(input, class_index, self._weight, reduction='none')
        if self._lambda is not None:
            B, S, H, W = error.size()

            w = torch.arange(1.0, 1.0 + S * self._lambda, self._lambda)
            if torch.cuda.is_available():
                w = w.to(error.get_device())
                # B, H, W, S
            error = (w * error.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # S*B*1*H*W
        error = error.permute(1, 0, 2, 3).unsqueeze(2)
        
        return torch.mean(error*mask.float())


class Weighted_mse_mae(nn.Module):
    def __init__(self, mse_weight=1.0, mae_weight=1.0, NORMAL_LOSS_GLOBAL_SCALE=0.00005, LAMBDA=None):
        super().__init__()
        self._NORMAL_LOSS_GLOBAL_SCALE = NORMAL_LOSS_GLOBAL_SCALE
        self._mse_weight = mse_weight
        self._mae_weight = mae_weight
        self._lambda = LAMBDA

    def forward(self, input, target, mask):
        balancing_weights = cfg.EVALUATION.BALANCING_WEIGHTS
        weights = torch.ones_like(input) * balancing_weights[0]
        thresholds = [dBZ_to_pixel(ele) for ele in cfg.EVALUATION.THRESHOLDS]

        for i, threshold in enumerate(thresholds):
            weights = weights + (balancing_weights[i+1] - balancing_weights[i]) * (target >= threshold).float()
        weights = weights * mask.float()

        # input: S*B*1*H*W
        # error: S*B
        mse = torch.sum(weights * ((input-target)**2), (2, 3, 4))
        mae = torch.sum(weights * (torch.abs((input-target))), (2, 3, 4))
        if self._lambda is not None:
            S, B = mse.size()
            w = torch.arange(1.0, 1.0 + S * self._lambda, self._lambda)
            if torch.cuda.is_available():
                w = w.to(mse.get_device())
            mse = (w * mse.permute(1, 0)).permute(1, 0)
            mae = (w * mae.permute(1, 0)).permute(1, 0)
        
        return self._NORMAL_LOSS_GLOBAL_SCALE * (self._mse_weight*torch.mean(mse) + self._mae_weight*torch.mean(mae))