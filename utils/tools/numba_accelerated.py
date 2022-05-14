import numpy as np
from numba import jit, float32, boolean, int32, float64
from utils.tools.evaluation import dBZ_to_pixel
from utils.config import cfg

@jit(int32(float32, float32, boolean, float32))
def _get_hit_miss_counts_numba(prediction, truth, mask, thresholds):
    seq_len, batch_size, _, height, width = prediction.shape
    threshold_num = len(thresholds)

    ret = np.zeros(shape=(seq_len, batch_size, threshold_num, 4), dtype=np.int32)

    for i in range(seq_len):
        for j in range(batch_size):
            for m in range(height):
                for n in range(width):
                    if mask[i][j][0][m][n]:
                        for k in range(threshold_num):
                            bpred = prediction[i][j][0][m][n] >= thresholds[k]
                            btruth = truth[i][j][0][m][n] >= thresholds[k]
                            ind = (1 - btruth) * 2 + (1 - bpred)
                            ret[i][j][k][ind] += 1
                            # The above code is the same as:
                            # TP
                            # ret[i][j][k][0] += bpred * btruth
                            # FP
                            # ret[i][j][k][1] += (1 - bpred) * btruth
                            # TN
                            # ret[i][j][k][2] += bpred * (1 - btruth)
                            # FN
                            # ret[i][j][k][3] += (1 - bpred) * (1- btruth)
    return ret

def get_hit_miss_counts_numba(prediction, truth, mask, thresholds=None):
    """
    This function calculates the overall hits and misses for the prediction, which could be used
    to get the skill scores and threat scores:


    This function assumes the input, i.e, prediction and truth are 3-dim tensors, (timestep, row, col)
    and all inputs should be between 0~1

    Parameters
    ----------
    prediction : np.ndarray
        Shape: (seq_len, batch_size, 1, height, width)
    truth : np.ndarray
        Shape: (seq_len, batch_size, 1, height, width)
    mask : np.ndarray or None
        Shape: (seq_len, batch_size, 1, height, width)
        0 --> not use
        1 --> use
    thresholds : list or tuple (optional)

    Returns
    -------
    hits : np.ndarray
        (seq_len, batch_size, len(thresholds))
        TP
    misses : np.ndarray
        (seq_len, batch_size, len(thresholds))
        FN
    false_alarms : np.ndarray
        (seq_len, batch_size, len(thresholds))
        FP
    true_negatives : np.ndarray
        (seq_len, batch_size, len(thresholds))
        TN
    """
    if thresholds is None:
        thresholds = cfg.EVALUATION.THRESHOLDS
    assert 5 == prediction.ndim
    assert 5 == truth.ndim
    assert prediction.shape == truth.shape
    assert prediction.shape[2] == 1
    thresholds = [dBZ_to_pixel(thresholds[i]) for i in range(len(thresholds))]
    thresholds = sorted(thresholds)
    ret = _get_hit_miss_counts_numba(prediction, truth, mask, thresholds)
    return ret[:, :, :, 0], ret[:, :, :, 1], ret[:, :, :, 2], ret[:, :, :, 3]


@jit(float32(float32, float32, boolean))
def get_GDL_numba(prediction, truth, mask):
    """
    Accelerated version of get_GDL using numba(http://numba.pydata.org/)

    Parameters
    ----------
    prediction : np.ndarray
        Shape: (seq_len, batch_size, 1, height, width)
    truth : np.ndarray
        Shape: (seq_len, batch_size, 1, height, width)
    mask : np.ndarray or None
        Shape: (seq_len, batch_size, 1, height, width)
        0 --> not use
        1 --> use

    Returns
    -------
    gdl : np.ndarray
        Shape: (seq_len,) or (seq_len, batch_size)
    """
    seq_len, batch_size, _, height, width = prediction.shape

    gdl = np.zeros(shape=(seq_len, batch_size), dtype=np.float32)

    for i in range(seq_len):
        for j in range(batch_size):
            for m in range(height):
                for n in range(width):
                    if m + 1 < height:
                        if mask[i][j][0][m+1][n] and mask[i][j][0][m][n]:
                            pred_diff_h = abs(prediction[i][j][0][m+1][n] - prediction[i][j][0][m][n])
                            gt_diff_h = abs(truth[i][j][0][m+1][n] - truth[i][j][0][m][n])
                            gdl[i][j] += abs(pred_diff_h - gt_diff_h)
                    if n + 1 < width:
                        if mask[i][j][0][m][n+1] and mask[i][j][0][m][n]:
                            pred_diff_w = abs(prediction[i][j][0][m][n+1] - prediction[i][j][0][m][n])
                            gt_diff_w = abs(truth[i][j][0][m][n+1] - truth[i][j][0][m][n])
                            gdl[i][j] += abs(pred_diff_w - gt_diff_w)
    return gdl


@jit(float32(float32, boolean, float32, float32))
def _get_balancing_weights_numba(data, mask, base_balancing_weights, thresholds):
    seq_len, batch_size, _, height, width = data.shape
    threshold_num = len(thresholds)
    ret = np.zeros(shape=(seq_len, batch_size, 1, height, width), dtype=np.float32)

    for i in range(seq_len):
        for j in range(batch_size):
            for m in range(height):
                for n in range(width):
                    if mask[i][j][0][m][n]:
                        ele = data[i][j][0][m][n]
                        for k in range(threshold_num):
                            if ele < thresholds[k]:
                                ret[i][j][0][m][n] = base_balancing_weights[k]
                                break
                        if ele >= thresholds[threshold_num - 1]:
                            ret[i][j][0][m][n] = base_balancing_weights[threshold_num]
    return ret

def get_balancing_weights_numba(data, mask, base_balancing_weights=None, thresholds=None):
    """Get the balancing weights
    
    Parameters
    ----------
    data
    mask
    base_balancing_weights
    thresholds

    Returns
    -------

    """
    if thresholds is None:
        thresholds = cfg.EVALUATION.THRESHOLDS
    if base_balancing_weights is None:
        base_balancing_weights = cfg.EVALUATION.BALANCING_WEIGHTS
    assert data.shape[2] == 1
    thresholds = [dBZ_to_pixel(thresholds[i]) for i in range(len(thresholds))]
    thresholds = sorted(thresholds)
    ret = _get_balancing_weights_numba(data, mask, base_balancing_weights, thresholds)
    return ret

# Move to numba_test.ipynb

# if __name__ == '__main__':
#     from utils.tools.evaluation import get_GDL, get_hit_miss_counts, get_balancing_weights
#     from numpy.testing import assert_allclose, assert_almost_equal
#     import time

#     prediction = np.random.uniform(size=(10, 16, 1, 480, 480))
#     truth = np.random.uniform(size=(10, 16, 1, 480, 480))
#     mask = np.random.randint(low=0, high=2, size=(10, 16, 1, 480, 480)).astype(bool)
    
#     begin = time.time()
#     gdl = get_GDL(prediction=prediction, truth=truth, mask=mask)
#     end = time.time()
#     print("numpy gdl:", end - begin)

#     begin = time.time()
#     gdl_numba = get_GDL_numba(prediction=prediction, truth=truth, mask=mask)
#     end = time.time()
#     print("numba gdl:", end - begin)

#     # gdl_mx = mx_get_GDL(prediction=prediction, truth=truth, mask=mask)
#     # print gdl_mx
#     assert_allclose(gdl, gdl_numba, rtol=1E-4, atol=1E-3)

#     begin = time.time()
#     for i in range(5):
#         hits, misses, false_alarms, true_negatives = get_hit_miss_counts(prediction, truth, mask)
#     end = time.time()
#     print("numpy hits misses:", end - begin)

#     begin = time.time()
#     for i in range(5):
#         hits_numba, misses_numba, false_alarms_numba, true_negatives_numba = get_hit_miss_counts_numba(prediction, truth, mask)
#     end = time.time()
#     print("numba hits misses:", end - begin)

#     print(np.abs(hits - hits_numba).max())
#     print(np.abs(misses - misses_numba).max(), np.abs(misses - misses_numba).argmax())
#     print(np.abs(false_alarms - false_alarms_numba).max(),
#           np.abs(false_alarms - false_alarms_numba).argmax())
#     print(np.abs(true_negatives - true_negatives_numba).max(),
#           np.abs(true_negatives - true_negatives_numba).argmax())

#     begin = time.time()
#     for i in range(5):
#         weights_npy = get_balancing_weights(data=truth, mask=mask,
#                                             base_balancing_weights=None, thresholds=None)
#     end = time.time()
#     print("numpy balancing weights:", end - begin)

#     begin = time.time()
#     for i in range(5):
#         weights_numba = get_balancing_weights_numba(data=truth, mask=mask,
#                                                     base_balancing_weights=None, thresholds=None)
#     end = time.time()
#     print("numba balancing weights:", end - begin)
#     print("Inconsistent Number:", (np.abs(weights_npy - weights_numba) > 1E-5).sum())
