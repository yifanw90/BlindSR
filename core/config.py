from easydict import EasyDict as edict
import numpy as np
import os

config = edict()

config.network = edict()
config.network.factor = 4
config.network.input_range = 1.0

config.test = edict()
config.test.batch_size = 1
config.test.model_path = 'models/'

config.test.data_path = 'images/'
config.test.data_set = 'Set5'
config.test.ker_id = 1

config.test.is_save = True
config.test.save_path = 'results/'

config.gpu_id = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id
config.gpu = ['cuda:%s'%i for i,j in enumerate(config.gpu_id.split(','))]
