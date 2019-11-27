import cv2
import numpy as np
import skimage.color as sc
from scipy import ndimage
from PIL import Image
import os

kernels = np.load('util/kernel_test.npy')

def load_img(im_name, factor):
    im = cv2.imread(im_name)
    h, w = im.shape[:2]
    h = h // factor * factor
    w = w // factor * factor
    im = im[:h, :w]
    return im

def rgb2ycbcr(im):
    return sc.rgb2ycbcr(im)

def ycbcr2rgb(im):
    im = sc.ycbcr2rgb(im)
    im = np.clip(im, 0, 1.0)*255.0
    return im.astype(np.uint8)

def generate_lr(im, factor, ker_id):
    h, w = im.shape[:2]
    kernel = kernels[..., 0, ker_id]
    im_blur = ndimage.correlate(im, kernel)
    im_lr = resize(im_blur, int(w/factor), int(h/factor), method=Image.NEAREST)
    im_bi = resize(im_lr, w, h)
    im = np.array(im, dtype=np.float32)
    im_lr = np.array(im_lr, dtype=np.float32)
    im_bi = np.array(im_bi, dtype=np.float32)
    return im, im_lr, im_bi


def resize(im, width, height, method=Image.BICUBIC):
    im = Image.fromarray(im)
    im = im.resize((width, height), method)
    return np.array(im)


def save_img(im_np, im_name, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    im = Image.fromarray(im_np)
    im.save(os.path.join(save_path, im_name))
