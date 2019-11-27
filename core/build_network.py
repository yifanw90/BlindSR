import torch
from torch.nn import DataParallel
from core.config import config
from network.net import Net


def build_network():
    net = Net(factor = config.network.factor, input_range = config.network.input_range)
    net = DataParallel(net, device_ids=config.gpu).to(config.gpu[0])
    return net
