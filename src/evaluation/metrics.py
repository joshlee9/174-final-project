import numpy as np
from skimage.metrics import structural_similarity as compare_ssim

def calculate_snr(reference, stitched):
    """
    Calculate Signal-to-Noise Ratio between reference and stitched images.
    """
    signal_power = np.sum(reference ** 2)
    noise_power = np.sum((reference - stitched) ** 2)
    if noise_power == 0:
        return float('inf')
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def calculate_ssim(reference, stitched):
    """
    Calculate Structural Similarity Index between reference and stitched images.
    """
    ssim_value, _ = compare_ssim(reference, stitched, full=True)
    return ssim_value

def calculate_rmse(reference, stitched):
    """
    Calculate Root Mean Square Error between reference and stitched images.
    """
    mse = np.mean((reference - stitched) ** 2)
    rmse = np.sqrt(mse)
    return rmse
