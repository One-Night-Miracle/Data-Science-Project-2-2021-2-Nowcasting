import numpy as np
from torch import nn
from collections import OrderedDict
from utils.config import cfg

import pandas as pd

### Block

def make_layers(block):
    layers = []
    for layer_name, v in block.items():

        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1],
                                    padding=v[2])
            layers.append((layer_name, layer))

        elif 'deconv' in layer_name:
            transposeConv2d = nn.ConvTranspose2d(in_channels=v[0], out_channels=v[1],
                                                 kernel_size=v[2], stride=v[3],
                                                 padding=v[4])
            layers.append((layer_name, transposeConv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name, nn.LeakyReLU(negative_slope=0.2, inplace=True)))

        elif 'conv' in layer_name:
            conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                               kernel_size=v[2], stride=v[3],
                               padding=v[4])
            layers.append((layer_name, conv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name, nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        
        elif 'interpolate' in layer_name:
            pass

        else:
            raise NotImplementedError

    return nn.Sequential(OrderedDict(layers))


### Conversion

"""
Default Parameter
----------
max_dBZ = 75.
a = 200.
b = 1.6
"""
_MAX_dBZ = cfg.GLOBAL.ZR.MAX_dBZ
_A = cfg.GLOBAL.ZR.A
_B = cfg.GLOBAL.ZR.B

def pixel_to_dBZ(img, max_dBZ=_MAX_dBZ):
    """

    Parameters
    ----------
    img : np.ndarray, float32
    max_dBZ : float32, optional

    Returns
    -------

    """
    return img * max_dBZ

def dBZ_to_pixel(dBZ_img, max_dBZ=_MAX_dBZ):
    """

    Parameters
    ----------
    dBZ_img : np.ndarray
    max_dBZ : float32, optional

    Returns
    -------

    """
    return np.clip(dBZ_img / max_dBZ, a_min=0.0, a_max=1.0)

def dBZ_to_rainfall(dBZ, a=_A, b=_B):
    """Convert the rainfall intensity to pixel values

    dBZ = dBa + b * dBR

    Parameters
    ----------
    rainfall_intensity : np.ndarray
    dBZ : np.ndarray, float32

    Returns
    -------
    rainfall_intensity : np.ndarray
    """
    return np.power(10, (dBZ * np.log10(a))/(10*b))

def rainfall_to_dBZ(rainfall, a=_A, b=_B):
    """Convert the rainfall intensity to pixel values

    dBZ = dBa + b * dBR

    Parameters
    ----------
    rainfall_intensity : np.ndarray
    dBZ : np.ndarray, float32

    Returns
    -------
    rainfall_intensity : np.ndarray
    """
    return 10*np.log10(a) + 10*b*np.log10(rainfall+0.5)

def pixel_to_rainfall(img, a=_A, b=_B):
    """Convert the pixel values to real rainfall intensity

    Z = aR^b

    Parameters
    ----------
    img : np.ndarray
    a : float32, optional
    b : float32, optional

    Returns
    -------
    rainfall_intensity : np.ndarray
    """
    dBZ = pixel_to_dBZ(img)
    rainfall_intensity = dBZ_to_rainfall(dBZ)
    return rainfall_intensity

def rainfall_to_pixel(rainfall_intensity, a=_A, b=_B):
    """Convert the rainfall intensity to pixel values

    dBZ = dBa + b * dBR

    Parameters
    ----------
    rainfall_intensity : np.ndarray
    a : float32, optional
    b : float32, optional

    Returns
    -------
    pixel_vals : np.ndarray
    """
    dBZ = rainfall_to_dBZ(rainfall_intensity)
    pixel_vals = dBZ_to_pixel(dBZ)
    return pixel_vals


def rebuild_bkk_pkl():
    filenames = pd.read_csv(cfg.ONM_CSV.ALL)[['DateTime','FolderPath','FileName']]
    # filenames = pd.read_csv(cfg.ONM_CSV.ALL)[['FileName', 'RADAR_RGB_PNG_PATH', 'RADAR_dBZ_PNG_PATH', 'RADAR_MASK_PATH']]
    filenames['DateTime'] = pd.to_datetime(filenames['DateTime'])
    filenames = filenames.set_index('DateTime')

    filenames['FileName'] = filenames['FileName'].str.replace(".png","")
    filenames['RADAR_RGB_PNG_PATH'] = filenames['FolderPath'].str.replace("/data/bkk_radar_images/bkk_radar_images_", "bkk_radar_images_")
    filenames['RADAR_RGB_PNG_PATH'] = filenames['RADAR_RGB_PNG_PATH'].str.replace(r"/", "", regex=True)
    filenames['RADAR_dBZ_PNG_PATH'] = filenames['FolderPath'].str.replace("/data/bkk_radar_images/bkk_radar_images_", "bkk_radar_images_dBZ_")
    filenames['RADAR_dBZ_PNG_PATH'] = filenames['RADAR_dBZ_PNG_PATH'].str.replace(r"/", "", regex=True)
    filenames['RADAR_MASK_PATH'] = filenames['FolderPath'].str.replace("/data/bkk_radar_images/bkk_radar_images_", "bkk_radar_images_mask_")
    filenames['RADAR_MASK_PATH'] = filenames['RADAR_MASK_PATH'].str.replace(r"/", "", regex=True)
    filenames = filenames.drop(columns=['FolderPath'])

    filenames.to_pickle(cfg.ONM_PD.FOLDER_ALL)

    for i in range(1,14):
        exec(f"filenames_{i} = filenames[filenames['RADAR_RGB_PNG_PATH']=='bkk_radar_images_{i}'].reset_index()[['FileName', 'RADAR_RGB_PNG_PATH', 'RADAR_dBZ_PNG_PATH', 'RADAR_MASK_PATH']]")
        exec(f"filenames_{i}.to_pickle(cfg.ONM_PD.FOLDER_{i})")