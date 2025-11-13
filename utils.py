import os
import numpy as np
import pandas as pd
from PIL import Image
import tifffile as tiff
from sklearn.preprocessing import LabelEncoder, StandardScaler


def read_image(path):
    try:
        return Image.open(path).convert('RGB')
    except:
        arr = tiff.imread(path)
        if arr.ndim == 2:
            arr = np.stack([arr]*3, -1)
        return Image.fromarray(arr.astype('uint8'))
