import numpy as np
from skimage.measure import compare_ssim
import math

def calc_psnr(gt, pre, scale, data_range=255.0, benchmark=True):
    pre = np.clip(pre, 0, data_range)
    gt = np.round(255*gt / data_range)
    pre = np.round(255*pre / data_range)
    diff = (gt - pre) / 255.0
    if benchmark:
        shave = scale
    else:
        shave = scale + 6

    valid = diff[shave:-shave, shave:-shave]
    mse = (valid**2).mean()
    if mse == 0:
        mse = 1e-6
    return -10 * math.log10(mse)


def calc_ssim(gt, pre, scale, data_range=255.0, benchmark=True):
    if benchmark:
        shave = scale
    else:
        shave = scale + 6
    pre = np.clip(pre, 0, data_range)
    gt = np.round(255*gt / data_range)
    pre = np.round(255*pre / data_range)
    gt = gt[shave:-shave, shave:-shave]
    pre = pre[shave:-shave, shave:-shave]
    ssim = compare_ssim(gt, pre, data_range=255.0)
    return ssim

