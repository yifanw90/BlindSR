import torch
import numpy as np
from core.config import config
from core.build_network import build_network
from core.compute_score import calc_psnr, calc_ssim
from core.sr import generate_sr
from util.util import *
from os.path import join
 
## image_name, blur kernel index, upscale factor 
im_name = 'baby.bmp'
factor = config.network.factor
ker_id =  config.test.ker_id

## read image and convert to ycbcr
im = load_img(join(config.test.data_path, config.test.data_set, im_name), factor)
im_ycbcr = sc.rgb2ycbcr(im[..., -1::-1])

## generate blur image
im_hr, im_lr, im_bi = generate_lr(im_ycbcr[..., 0], factor, ker_id) 

## load models and run SR 
net = build_network()
ckpt = torch.load(join(config.test.model_path, 'epoch-x%d.pth'%factor))
net.load_state_dict(ckpt['model_state_dict'])
im_sr = generate_sr(net, im_lr, im_bi, config)
    
## compute scores on Y channel
psnr = calc_psnr(im_hr/255.0, im_sr, factor, data_range = config.network.input_range, benchmark=True)
ssim = calc_ssim(im_hr/255.0, im_sr, factor, data_range=config.network.input_range, benchmark=True)
print('Img: %s, factor: %d, psnr: %.2f, ssim: %.4f' %(im_name, factor, psnr, ssim))
    
## save rgb SR results 
if config.test.is_save:
    im_sr = np.clip(im_sr, 0, 1.0)*255.0
    ## super-resolve cb, cr channles using bicubic
    _, _, im_cb = generate_lr(im_ycbcr[..., 1], factor, ker_id)
    _, _, im_cr = generate_lr(im_ycbcr[..., 2], factor, ker_id)
    im_sr_rgb = np.stack([im_sr, im_cb, im_cr], axis=-1)
    im_sr_rgb = ycbcr2rgb(im_sr_rgb)
    save_img(im_sr_rgb, im_name.split('.')[0]+'_sr_x%d.png'%factor, config.test.save_path)
