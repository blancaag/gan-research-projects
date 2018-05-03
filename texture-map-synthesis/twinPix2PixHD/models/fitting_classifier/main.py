from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import models, transforms
# import matplotlib.pyplot as plt
import time
import os
from PIL import Image
# import importlib.util
# from importlib import reload
from imp import reload
import argparse

import DatasetFromSubFolders
from DatasetFromSubFolders import *

# from . import networks
import networks

from options import Options
from train import *
from test import *

# for loading Python 2 generated model-files
from functools import partial
import pickle
pickle.load = partial(pickle.load, encoding="latin1")
pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")

use_gpu = torch.cuda.is_available()
print('Num. of cuda visible devices: %d' %torch.cuda.device_count(), list(range(torch.cuda.device_count())))   

# Loading options
opt = Options().parse()

# Setting the data loaders
from data import *

phases = ['train', 'val']
datasets, dataloaders = set_dataloaders(opt.data_dir, opt.batch_size, opt.nworkers, opt.target_size, opt.phases)
datasets_sizes = {x: len(datasets[x]) for x in phases}
print(datasets_sizes)

# Setting the model
# model = DenseNetMulti(nchannels=opt.nc_input)

use_gpu = torch.cuda.is_available()

model = networks.DenseNetMulti(nchannels=opt.nc_input)
model.initialize(opt)
if use_gpu: model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count()))).cuda()

pretrained = 'checkpoints_densenet_test/test_clean/model_best_c4_0.742152_0.018293_1519316.pt'      
pretrained = 'checkpoints_densenet_test/test_clean/model_best_c4_0.829596_0.023838_1519321.pt'   

# uncomment below if  want to add pretrained   
checkpoint = None; #pretrained
# if torch.cuda.device_count() == 1:
if checkpoint: model.load_state_dict(torch.load(checkpoint + 'h.tar'))
# model = torch.load(checkpoint); print(model)



# use_gpu = len(gpu_ids) > 0
# norm_layer = get_norm_layer(norm_type=norm)


# model2 = DenseNetMulti(nchannels=opt.nc_input)
# model3 = DenseNetMulti(nchannels=opt.nc_input)
# model4 = DenseNetMulti(nchannels=opt.nc_input)

# num_ftrs = model.classifier.in_features
# model.classifier = nn.Linear(num_ftrs, 2)


# # device_ids=list(range(torch.cuda.device_count()))
# if use_gpu: model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count()))).cuda()
#
# """ Load model """
# # checkpoint = os.path.join(checkpoint_dir_, 'model_best_c4_0.753670.pt')
# # if opt.checkpoint: model = torch.load(opt.checkpoint, map_location={'cuda:0':'cuda:1'}) #map_location=lambda storage, loc: storage.cuda())
# if opt.checkpoint:
#         # model = torch.load(opt.checkpoint)
#         model_dir = os.path.join(opt.checkpoint_dir, opt.name)
#         if torch.cuda.device_count() == 1: model.load_state_dict(torch.load(opt.checkpoint))
#         else : model = torch.load(opt.checkpoint) #, map_location={'cuda:0':'cuda:1'}) #map_location=lambda storage,            loc: storage.cuda())
#         # model.load_state_dict(torch.load(opt.checkpoint))
#
# # checkpoint = 'model_best_c4_75.2.pt'
# # model = torch.load(checkpoint, map_location=lambda storage, loc: storage, pickle_module=pickle)
# """ """

criterion1 = nn.CrossEntropyLoss()
criterion2 = nn.BCELoss()
criterion3 = nn.BCEWithLogitsLoss()
criterion4 = nn.MSELoss()

if opt.phase == 'train':
        model = train_model(model, dataloaders, criterion3, opt, datasets_sizes)
                            
        # model2 = train_model(model2,
        #                     dataloaders, criterion2, optimizer, scheduler, opt.nc_input, opt.nepochs, opt.phases,
        #                                         opt.checkpoint_dir)
        # model3 = train_model(model3,
        #                      dataloaders, criterion3, optimizer, scheduler, opt.nc_input, opt.nepochs, opt.phases,
        #                                                             opt.checkpoint_dir)
        # model4 = train_model(model4,
        #                     dataloaders, criterion4, optimizer, scheduler, opt.nc_input, opt.nepochs, opt.phases,
        #                     opt.checkpoint_dir)
else: 
        print('Add test code from notebook')
        
# for group in optimizer.param_groups:
#     print (group['lr'])
# #     group['lr'] = 0.0001
#     print (group['lr'])
#
#
# ### TESTING ###
# phases = ['test']
# data_dir = '/blanca/datasets/2nd_FLLC_MB/output/trainset/preprocessed_ilstack'
# data_dir = '.../datasets/2nd_FLLC_MB/output/trainset/preprocessed_ilstack'
#
# if opt.test_data: data_dir = opt.test_data
#
# datasets, dataloaders = set_dataloaders(data_dir, batch_size, nworkers, phases)
#
# test_model(model, dataloaders, phases) #### add output folders inside output