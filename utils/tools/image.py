# Python plugin that supports loading batch of images in parallel
import cv2
import numpy as np
import threading
import os
import struct
from utils.config import cfg
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, wait

_imread_executor_pool = ThreadPoolExecutor(max_workers=cfg.GLOBAL.MAX_WORKERS)

img_width = cfg.ONM.ITERATOR.WIDTH
img_height = cfg.ONM.ITERATOR.HEIGHT

def cv2_read_img(path, read_storage, grayscale=True, resize_storage=None, frame_size=None):
    """
    Load mask file (image) to numpy array
    (optional: resize image)

    Parameters
    ----------
    path : str
    read_storage : np.ndarray
    grayscale : bool, optional
    resize_storage : np.ndarray, optional
    frame_size : list or tuple, optional

    Returns
    -------
    """
    if frame_size is None:
        frame_size = (img_height, img_width)
    
    if grayscale:
        read_storage[:] = cv2.imread(path, 0)
    else:
        read_storage[:] = cv2.imread(path)
    
    if resize_storage is not None:
        resize_storage[:] = cv2.resize(read_storage, frame_size, interpolation=cv2.INTER_LINEAR)


def quick_read_frames(path_list, resize=False, frame_size=None, grayscale=True):
    """
    Multi-thread Frame Loader
    (Load mask files to numpy array)

    Parameters
    ----------
    path_list : list
    resize : bool, optional
    frame_size : None or tuple, optional
    grayscale : bool, optional

    Returns
    -------
    resize_storage/read_storage : np.ndarray
    """
    img_num = len(path_list)

    for i in range(img_num):
        if not os.path.exists(path_list[i]):
            raise IOError
    
    if frame_size is None:
        frame_size = (img_height, img_width)
    im_h, im_w = frame_size

    if grayscale:
        read_storage = np.empty((img_num, im_h, im_w), dtype=np.uint8)
    else:
        read_storage = np.empty((img_num, im_h, im_w, 3), dtype=np.uint8)
    
    ### Resize
    if resize:
        if grayscale:
            resize_storage = np.empty((img_num, im_h, im_w), dtype=np.uint8)
        else:
            resize_storage = np.empty((img_num, im_h, im_w, 3), dtype=np.uint8)

        ## One Image ##
        if img_num == 1:
            cv2_read_img(path_list[0], read_storage[0], grayscale, resize_storage[0], frame_size)

        ## Many Images ##
        else:
            future_objs = []
            for i in range(img_num):
                obj = _imread_executor_pool.submit(cv2_read_img, path_list[i], read_storage[i], grayscale, resize_storage[i], frame_size)
                future_objs.append(obj)
            wait(future_objs)

        if grayscale:
            resize_storage = resize_storage.reshape((img_num, 1, im_h, im_w))
        else:
            resize_storage = resize_storage.transpose((0, 3, 1, 2))

        return resize_storage[:, ::-1, ...]
    
    ### NOT Resize ###
    else:
        ## One Image ##
        if img_num == 1:
            cv2_read_img(path_list[0], read_storage[0], grayscale)
        
        ## Many Images ##
        else:
            future_objs = []
            for i in range(img_num):
                obj = _imread_executor_pool.submit(cv2_read_img, path_list[i], read_storage[i], grayscale)
                future_objs.append(obj)
            wait(future_objs)
        
        if grayscale:
            read_storage = read_storage.reshape((img_num, 1, im_h, im_w))
        else:
            read_storage = read_storage.transpose((0, 3, 1, 2))

        return read_storage[:, ::-1, ...]