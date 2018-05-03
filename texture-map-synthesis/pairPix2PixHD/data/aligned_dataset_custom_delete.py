### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os.path
# from data.base_dataset import BaseDataset, get_params, get_transform, normalize, fill_gaps, read_image_OpenCV
# from data.image_folder import make_dataset
import torchvision.transforms as transforms
import torch.utils.data as data

import numpy as np
import random
from pathlib import Path
from functools import reduce
import operator
import cv2
from PIL import Image

def read_image_PIL(path, opt):
        im = Image.open(path).convert('RGB')
        im = im.resize((opt.loadSize * 2, opt.loadSize), Image.BICUBIC)
        return im

def read_image_OpenCV(path, opt, is_pair=False):
        im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if is_pair: im = cv2.resize(im, (opt.loadSize * 2, opt.loadSize))
        else: im = cv2.resize(im, (opt.loadSize, opt.loadSize))
        return im

# def reduce_and_shuffle_dict_values_nested1level(d):
#
#     flat = reduce(operator.add,
#                       [reduce(operator.add, i.values(), []) for i in list(d.values())],
#                       [])
#
#     [random.shuffle(flat) for i in range(int(1e4))]
#
#     return flat

# def create_dataset_from_dir2subdir(dir, nitems=None):
#
#     """
#     Create a list of paths from a nesteed dir with two levels, selecting nitems from each dir of the last level
#     """
#
#     EXT_RECURSIVE = ['**/*.jpg', '**/*.JPG', '**/*.png', '**/*.ppm']
#     from collections import OrderedDict
#
#     path = Path(dir)
#     id_names = [i.parts[-1] for i in list(path.glob('*')) if os.path.isdir(i)]
#
#     n_items_per_last_level = nitems
#
#     data_dict = OrderedDict({i: {} for i in sorted(id_names)})
#     data_dict_nitems = OrderedDict({i: {} for i in sorted(id_names)})
#
#     # INITIALISE
#     for i in id_names:
#         for j in os.listdir(path/i):
#             data_dict[i][j] = None
#
#     # FILLING
#     import random
#     random.seed()
#
#     for i in data_dict.keys():
#         for j in data_dict[i].keys():
#             txt_pl = reduce(
#                   operator.add,
#                   [list((path/i/j).glob('**/*.isomap.png'))],
#                   [])
#
#             # DICT WITH ALL PATHS
#             data_dict[i][j] = txt_pl
#
#             # DICT WITH MAX(N) PATHS
#             random_idx = random.sample(range(len(txt_pl)), min(len(txt_pl), n_items_per_last_level))
#             txt_pl_nitems = [str(txt_pl[i]) for i in random_idx]
#
#             data_dict_nitems[i][j] = txt_pl_nitems
#
#     print('Total found IDs in path %s: %d' %(path, len(data_dict_nitems)), '.. and selected %d per ID' %n_items_per_last_level)
#
#     data_list_n_shuffled = reduce_and_shuffle_dict_values_nested1level(data_dict_nitems)
#
#     return data_list_n_shuffled
#
# def make_dataset_fromIDsubfolders(dir, nitems=None):
#     assert os.path.isdir(dir), '%s is not a valid directory' % dir
#     images = create_dataset_from_dir2subdir(dir, nitems)
#     return images

def fill_gaps(im, opt,
              fill_input_with=None,
              add_artificial=False,
              only_artificial=False,
              cm_p = '/blanca/resources/contour_mask/contour_mask.png'):
    
    if not fill_input_with: fill_input_with = opt.fill
    
    # creating fills for filling gaps
    n_fill_WB = cv2.randu(np.zeros(im.shape[:2]), 0, 255)
    n_fill_B = np.zeros(im.shape[:2]) * 255
    n_fill_W = np.ones(im.shape[:2]) * 255
    n_fill_G = np.ones(im.shape[:2]) * 127
    
    which_fill = dict(zip(['W', 'B', 'W&B', 'G'], [n_fill_W, n_fill_B, n_fill_WB, n_fill_G]))

    # read contour template mask
    # im_cm = cv2.imread(cm_p, cv2.IMREAD_UNCHANGED)
    im_cm = read_image_OpenCV(cm_p, opt, is_pair=False)
    assert im_cm is not None, 'Make sure there is a "contour_mask.png" file in this folder'
    mask = im_cm == 255 # contour mask: internal 
    
    new_alpha = im[:,:,3] != 0
    new_alpha[im_cm == 0] = 1 # ensuring we exclude corners what != txt. map
    im[:,:,3] = new_alpha * 255 # applying
    
    if not only_artificial:
        for i in range(im.shape[2] - 1):
            if fill_input_with=='average':
                mask_average = im[:,:,3] != 0
                # print('Filling gaps with %f instead of %f' %(np.mean(im[:,:,0]), np.mean(im[:,:,0][mask_average])))
                im[:,:,i][~new_alpha] = (np.ones(im.shape[:2]) * np.mean(im[:,:,i][mask_average]))[~new_alpha]    
            else: im[:,:,i][~new_alpha] = which_fill[fill_input_with][~new_alpha]
    
    if add_artificial and opt.phase != 'test':
        ## SELECT AN OCCLUSSION AND APPLY TO THE IMAGE
        gpath = Path('/blanca')
        rpath = gpath / 'resources/db_occlusions'
        rpath = [rpath]
        rpaths = reduce(operator.add, 
                      [list(j.glob('*')) for j in rpath],
                      [])
        
        random_idx = random.sample(range(len(rpaths)), len(rpaths))
        ix = random.sample(range(len(rpaths)), 1)[0]
        # if opt.phase == 'test':
        #         rpath_test = '/blanca/resources/db_oclusions_test/000220_39.png'
        #         imr = read_image_OpenCV(rpath_test, opt, is_pair=False)
        # else:
        imr = read_image_OpenCV(str(rpaths[ix]), opt, is_pair=False)
        # imr = cv2.imread(str(rpaths[ix]), cv2.IMREAD_UNCHANGED)
        alpha_artifitial = imr
        alpha_artifitial[alpha_artifitial != 0] = 1
        im[:,:,3][im[:,:,3] != 0] = 1
        new_alpha_artifitial = alpha_artifitial * im[:,:,3]
        # setting the new alpha channel
        im[:,:,3] = new_alpha_artifitial * 255
        
        # filling
        for i in range(im.shape[2] - 1):
            im[:,:,i][new_alpha_artifitial == 0] = which_fill[fill_input_with][new_alpha_artifitial == 0]
    
    return im

def create_dataset(path, return_pairs=None):
        path = Path(path)
        ims_path_list = path.glob('*_m.png')
        pairs_path_list = []
        for i in ims_path_list:
                pair_name = i.parts[-1].split('_m.png')[0] + '.png'
                pair_path = path.glob(pair_name)
                pair_path = str(list(pair_path)[0])
                pair_path_mirror = str(i)
                pairs_path_list.append([pair_path, pair_path_mirror])

        random.seed(1984)
        [random.shuffle(pairs_path_list) for i in range(int(1e4))]
        
        if return_pairs==None: pairs_path_list = reduce(operator.add, pairs_path_list, [])
        return pairs_path_list


def apply_data_transforms(im, which, opt, nchannels):
        
    transform_list = []
    if which == 'target':
        transform_list += [
                transforms.Lambda(lambda x: fill_gaps(x, opt, fill_input_with='average')),
                transforms.Lambda(lambda x: x[:, :, :nchannels]),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
    elif which == 'input':        
        transform_list += [
            transforms.Lambda(lambda x: x.copy()),
            transforms.Lambda(lambda x: fill_gaps(x, opt, add_artificial=opt.isTrain)),
            transforms.Lambda(lambda x: x[:, :, :nchannels]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]                                    

    return transforms.Compose(transform_list)(im)
    
    
class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    
        self.target_paths = create_dataset(self.root, return_pairs=True)
        self.dataset_size = len(self.target_paths) 
      
    def __getitem__(self, index):                              
        
        target_tensor = target_tensor_mirror = inst_tensor = feat_tensor = 0
        
        input_nc = self.opt.label_nc if self.opt.label_nc != 0 else 3
        output_nc = self.opt.output_nc
        
        # read target image
        target_path = self.target_paths[index][0]
        target_path_mirror = self.target_paths[index][1]
        
        target_im = read_image_OpenCV(target_path, self.opt)
        target_im_mirror = read_image_OpenCV(target_path_mirror, self.opt)
        
        # create input tensor first
        # if self.opt.isTrain:
        input_tensor = apply_data_transforms(target_im, 'input', self.opt, input_nc)
        input_tensor_mirror = apply_data_transforms(target_im_mirror, 'input', self.opt, input_nc)
        
        # create output tensor
        if self.opt.isTrain:
            target_tensor = apply_data_transforms(target_im, 'target', self.opt, output_nc)
            target_tensor_mirror = apply_data_transforms(target_im_mirror, 'target', self.opt, output_nc)
         
        input_dict = {'label': [input_tensor, input_tensor_mirror], 'inst': inst_tensor, 
                      'image': [target_tensor, target_tensor_mirror], 'feat': feat_tensor, 
                      'path': [target_path, target_path_mirror]}

        return input_dict

    def __len__(self):
        return len(self.target_paths)

    def name(self):
        return 'AlignedDataset'