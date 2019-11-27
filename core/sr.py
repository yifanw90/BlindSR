import torch
import numpy as np
 
def generate_sr(net, im_lr, im_bi, config):
    im_lr = torch.from_numpy(im_lr[None, None])/255.0
    im_bi = torch.from_numpy(im_bi[None, None])/255.0

    im_lr = im_lr.pin_memory().to(config.gpu[0])
    im_bi = im_bi.pin_memory().to(config.gpu[0])

    net.eval()
    with torch.no_grad():
        im_sr = net(im_lr, im_bi)
        im_sr_np = im_sr.detach().cpu().numpy()
    
    return im_sr_np[0, 0] 
