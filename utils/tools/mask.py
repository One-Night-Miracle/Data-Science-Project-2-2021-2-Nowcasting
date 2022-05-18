# File to deal read and write the .mask extensions
import zlib
import numpy as np
from utils.config import cfg
from concurrent.futures import ThreadPoolExecutor, wait

_imread_executor_pool = ThreadPoolExecutor(max_workers=16)

img_width = cfg.ONM.ITERATOR.WIDTH
img_height = cfg.ONM.ITERATOR.HEIGHT

def read_mask_file(filepath, out=None):
    """
    Load mask file to numpy array

    Parameters
    ----------
    filepath : str
    out : np.ndarray

    Returns
    out : np.ndarray (optional)
    -------

    """
    f = open(filepath, 'rb')
    dat = zlib.decompress(f.read())

    if out is None:
        return np.frombuffer(dat, dtype=int).reshape((img_width, img_height)).astype(bool)
    
    out[:] = np.frombuffer(dat, dtype=int).reshape((img_width, img_height)).astype(bool)
    f.close()


def save_mask_file(npy_mask, filepath):
    """
    Save numpy array to mask file

    Parameters
    ----------
    npy_mask : np.ndarray
    filepath : str

    Returns
    -------

    """
    # compress level = 2
    compressed_data = zlib.compress(npy_mask.tobytes(), 2)
    f = open(filepath, "wb")
    f.write(compressed_data)
    f.close()


def quick_read_masks(path_list):
    """
    Quick load mask file to numpy array
    (with Threading Utilization)

    Parameters
    ----------
    filepath : str
    out : np.ndarray

    Returns
    -------
    masks : np.ndarray

    """
    num = len(path_list)
    read_storage = np.empty((num, img_width, img_height), dtype=np.bool)

    future_objs = []
    for i in range(num):
        obj = _imread_executor_pool.submit(read_mask_file, path_list[i], read_storage[i])
        future_objs.append(obj)
    wait(future_objs)

    masks = read_storage.reshape((num, 1, img_width, img_height))
    
    return masks