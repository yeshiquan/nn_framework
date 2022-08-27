import os
import numpy as np

def zero_pad(X, pad_width, dims):
    dims = (dims) if isinstance(dims, int) else dims
    pad = [(0,0) if idx not in dims else (pad_width, pad_width) for idx in range(len(X.shape))]
    X_padded = np.pad(X, pad, "constant")
    return X_padded