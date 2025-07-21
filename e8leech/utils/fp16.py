import numpy as np

def to_fp16(arr):
    """
    Converts a numpy array to fp16.
    """
    return arr.astype(np.float16)

def from_fp16(arr):
    """
    Converts a numpy array from fp16 to fp32.
    """
    return arr.astype(np.float32)
