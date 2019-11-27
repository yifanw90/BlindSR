import torch
import numpy as np
from core.config import config
from core.build_network import build_network
from core.compute_score import calc_psnr, calc_ssim
from core.sr import generate_sr
from util.util import *
from os.path import join
 
factor = config.network.factor

net = build_network()
ckpt = torch.load(join(config.test.model_path, 'epoch-x%d.pth'%factor))
net.load_state_dict(ckpt['model_state_dict'])

print('Processing dataset %s on all unseen blur kernels, upscale factor %d' %(config.test.data_set, factor))

img_names = os.listdir(join(config.test.data_path, config.test.data_set))

psnr, ssim = 0.0, 0.0

for i, im_name in enumerate(img_names):
    im = load_img(join(config.test.data_path, config.test.data_set, im_name), factor)
    im_ycbcr = sc.rgb2ycbcr(im[..., -1::-1])
    
    for ker_id in range(kernels.shape[-1]):
        im_hr, im_lr, im_bi = generate_lr(im_ycbcr[..., 0], factor, ker_id) 
        im_sr = generate_sr(net, im_lr, im_bi, config)
        
        ## compute scores
        psnr += calc_psnr(im_hr/255.0, im_sr, factor, data_range = config.network.input_range, benchmark=True)
        ssim += calc_ssim(im_hr/255.0, im_sr, factor, data_range=config.network.input_range, benchmark=True)

img_num = len(img_names) * kernels.shape[-1]
print('Average PSNR: %.2f, SSIM: %.4f' %(psnr/img_num, ssim/img_num))
