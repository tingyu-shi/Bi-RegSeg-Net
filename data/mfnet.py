#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time: 2024/12/11 16:42
# @Author: Tingyu Shi
# @File: mfnet.py
# @Description: mfnet
import glob
import os

import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
from torchvision import transforms
from data.augmentations import Compose, ColorJitter, RandomCrop, RandomScale, RandomFlip


class MFNet_Dataset(data.Dataset):
    def __init__(self, cfg, mode='test', do_aug=False):
        """
        Initializes the dataset with configurations and augmentation settings.

        Args:
            cfg (dict): Configuration dictionary containing dataset settings.
            do_aug (bool, optional): Whether to perform data augmentation. Defaults to False.
        """
        # Initialize name, mode and do_aug from the configuration and arguments
        self.name = cfg['dataset']
        self.mode = mode
        assert self.mode in ['train', 'val', 'test'], f'{self.mode} not support.'
        self.do_aug = do_aug

        ## pre-processing
        # Transformations for image data
        if cfg['normalize']:
            self.im_to_tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                # transforms.Normalize([0.23584902, 0.26904956, 0.23294316], [0.23274839, 0.23397096, 0.23796114])
            ])
            # Transformations for thermal data
            self.the_to_tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.449, 0.449, 0.449], [0.226, 0.226, 0.226])
                # transforms.Normalize([0.34428293, 0.34428293, 0.34428293], [0.16110095, 0.16110095, 0.16110095])
            ])
        else:
            self.im_to_tensor = transforms.Compose([
                transforms.ToTensor()
            ])
            self.the_to_tensor = transforms.Compose([
                transforms.ToTensor()
            ])

        # Root directory and number of classes from configuration
        self.root = cfg['root']
        self.n_classes = cfg['n_classes']

        if mode == 'train' and self.do_aug:
            # # Parse scale range and crop size from configuration for data augmentation
            # scale_range = tuple(float(i) for i in cfg['scales_range'].split(' '))
            # crop_size = tuple(int(i) for i in cfg['crop_size'].split(' '))

            # Data augmentation pipeline
            self.aug = Compose([
                # ColorJitter(
                #     brightness=cfg['brightness'],
                #     contrast=cfg['contrast'],
                #     saturation=cfg['saturation']),
                # RandomHorizontalFlip(cfg['p']),
                # RandomScale(scale_range),
                # RandomCrop(crop_size, pad_if_needed=True),
                RandomFlip(prob=cfg['flip_prob']),
                # RandomCrop(crop_rate=cfg['crop_rate'], prob=cfg['crop_prob']),
            ])

            # self.val_resize = Resize(crop_size)

        # Class weights for loss computation, based on class_weight setting in configuration
        if cfg['class_weight'] == 'enet':
            self.class_weight = np.array(
                [1.50739254, 16.76961434, 31.76282196, 34.79953093, 39.13100502, 41.31683502, 8.23649092, 46.36553788,
                 44.1501495])
        elif cfg['class_weight'] == 'median_freq_balancing':
            self.class_weight = np.array(
                [0.02673469, 0.34660353, 1.66635563, 1., 1.91559195, 1.29478313, 0.6958637, 2.99703005, 0.60996919])
        else:
            raise (f"{cfg['class_weight']} not support.")

        # Load all images' name from a text file based on the mode
        with open(os.path.join(self.root, f'{self.mode}.txt'), 'r') as f:
            self.infos = f.readlines()
            self.infos = [info for info in self.infos if not info.endswith('_flip\n')]


    def __len__(self):
        return len(self.infos)

    def __getitem__(self, index):
        image_name = self.infos[index].strip()

        vis = Image.open(os.path.join(self.root, 'pad', 'vis', image_name + '.png'))
        ir_pseudo = Image.open(os.path.join(self.root, 'pad', 'ir_pseudo_stylebooth', image_name + '.png'))
        ir_pseudo = ir_pseudo.convert('RGB')
        label = Image.open(os.path.join(self.root, 'labels', image_name + '.png'))
        ir = Image.open(os.path.join(self.root, 'pad', 'ir', image_name + '.png'))
        ir = ir.convert('RGB')
        ir_affine_deform = Image.open(os.path.join(self.root, 'pad', 'ir_affine_deform', image_name + '.png'))
        ir_affine_deform = ir_affine_deform.convert('RGB')

        sample = {
            'vis': self.im_to_tensor(vis),                                       # [b, c, h, w]
            'ir_pseudo': self.im_to_tensor(ir_pseudo),                           # [b, c, h, w]
            'ir': self.im_to_tensor(ir),                                         # [b, c, h, w]
            'ir_affine_deform': self.im_to_tensor(ir_affine_deform),             # [b, c, h, w]
            'label': torch.from_numpy(np.asarray(label, dtype=np.int64)).long()  # [b, h, w]
        }

        if self.mode == 'train' and self.do_aug:
            sample = self.aug(sample)

        return image_name, sample

    @property
    def cmap(self):
        return [
            (0, 0, 0),        # 0 unlabelled
            (64, 0, 128),     # 1 car
            (64, 64, 0),      # 2 person
            (0, 128, 192),    # 3 bike
            (0, 0, 192),      # 4 curve
            (128, 128, 0),    # 5 car_stop
            (64, 64, 128),    # 6 guardrail
            (192, 128, 128),  # 7 color_cone
            (192, 64, 0)      # 8 bump
        ]

