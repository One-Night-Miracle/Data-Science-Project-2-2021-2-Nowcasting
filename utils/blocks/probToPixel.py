import torch
from utils.config import cfg
from utils.utils import dBZ_to_pixel
import numpy as np

class ProbToPixel(object):

    def __init__(self, middle_value, requires_grad=False, NORMAL_LOSS_GLOBAL_SCALE=0.00005):
        if requires_grad:
            self._middle_value = torch.from_numpy(middle_value, requires_grad=True)
        else:
            self._middle_value = torch.from_numpy(middle_value)
        self.requires_grad = requires_grad
        self.NORMAL_LOSS_GLOBAL_SCALE = NORMAL_LOSS_GLOBAL_SCALE

    def __call__(self, prediction, ground_truth, mask, lr):
        # prediction: S*B*C*H*W
        result = np.argmax(prediction, axis=2)[:, :, np.newaxis, ...]
        prediction_result = np.zeros(result.shape, dtype=np.float32)
        if not self.requires_grad:
            for i in range(len(self._middle_value)):
                prediction_result[result==i] = self._middle_value[i]

        else:
            balancing_weights = cfg.EVALUATION.BALANCING_WEIGHTS
            weights = torch.ones_like(prediction_result) * balancing_weights[0]
            thresholds = cfg.EVALUATION.THRESHOLDS

            for i, threshold in enumerate(thresholds):
                weights = weights + (balancing_weights[i + 1] - balancing_weights[i]) * (ground_truth >= threshold).float()
            weights = weights * mask.float()


            loss = torch.zeros(1, requires_grad=True).float()
            for i in range(len(self._middle_value)):
                m = (result == i)
                prediction_result[m] = self._middle_value.data[i]
                tmp = (ground_truth[m]-self._middle_value[i])
                mse = torch.sum(weights[m] * (tmp ** 2), (2, 3, 4))
                mae = torch.sum(weights[m] * (torch.abs(tmp)), (2, 3, 4))
                loss = self.NORMAL_LOSS_GLOBAL_SCALE * (torch.mean(mse) + torch.mean(mae))
            loss.backward()
            self._middle_value -= lr * self._middle_value.grad

        return prediction_result
