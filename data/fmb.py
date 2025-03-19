#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time: 2024/12/6 19:43
# @Author: Tingyu Shi
# @File: fmb.py
# @Description: fmb
import os
import glob

import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
from torchvision import transforms
from data.augmentations import Compose, ColorJitter, RandomCrop, RandomScale, RandomFlip


class FMB_Dataset(data.Dataset):
    def __init__(self, cfg, mode='test', do_aug=False):
        """
        Initialize the dataset.

        Args:
            cfg (dict): Configuration dictionary, containing dataset configuration information.
            do_aug (bool): Whether to perform data augmentation. Default is False.

        Raises:
            AssertionError: If the mode in cfg is not 'train' or 'test'.
            ValueError: If the class_weight in cfg is not a supported option.
        """
        # Initialize name, mode and do_aug from the configuration and arguments
        self.name = cfg['dataset']
        self.mode = mode
        assert self.mode in ['train', 'test'], f'{self.mode} not support.'
        self.do_aug = do_aug

        ## pre-processing
        # Image preprocessing, convert image to tensor and normalize
        if cfg['normalize']:
            self.im_to_tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                # transforms.Normalize([0.47112003, 0.46897992, 0.45965695], [0.17483631, 0.17371644, 0.18179253])
            ])
            # Thermal image preprocessing, convert image to tensor and normalize
            self.the_to_tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.449, 0.449, 0.449], [0.226, 0.226, 0.226])
                # transforms.Normalize([0.30886722, 0.30886722, 0.30886722], [0.19067425, 0.19067425, 0.19067425])
            ])
        else:
            self.im_to_tensor = transforms.Compose([
                transforms.ToTensor()
            ])
            self.the_to_tensor = transforms.Compose([
                transforms.ToTensor()
            ])
        # Path to dataset root directory
        self.root = os.path.join(cfg['root'], 'c' + self.mode)
        # Number of classes
        self.n_classes = cfg['n_classes']

        if mode == 'train' and self.do_aug:
            # # Parse configuration information for data augmentation
            # scale_range = tuple(float(i) for i in cfg['scales_range'].split(' '))
            # crop_size = tuple(int(i) for i in cfg['crop_size'].split(' '))

            # Data augmentation
            self.aug = Compose([
                # ColorJitter(
                #     brightness=cfg['brightness'],
                #     contrast=cfg['contrast'],
                #     saturation=cfg['saturation']),
                # RandomHorizontalFlip(cfg['p']),
                # RandomScale(scale_range),
                # RandomCrop(crop_size, pad_if_needed=True),
                RandomFlip(prob=cfg['flip_prob']),
                RandomCrop(crop_rate=cfg['crop_rate'], prob=cfg['crop_prob']),
            ])

        # Class weights for loss calculation
        if cfg['class_weight'] == 'enet':
            self.class_weight = np.array(
                [26.55246552, 7.1305325, 28.44889292, 6.07373698, 47.8163219, 44.31616079, 4.33855057, 3.15897955,
                 42.42061434, 14.23343092, 42.72377203, 41.52787295, 49.37122801, 50.46671947, 28.5996613])
        elif cfg['class_weight'] == 'median_freq_balancing':
            self.class_weight = np.array(
                [1.30801361, 0.17981984, 0.99921909, 0.13529038, 6.68967884, 3.88262428, 0.09914717, 0.06789992,
                 4.49684873, 0.43160972, 0.9508358, 1., 6.33168409, 15.29381398, 1.41841193])
        else:
            raise (f"{cfg['class_weight']} not support.")

        # Get all images' path
        self.infos = glob.glob(os.path.join(self.root, 'pad', 'vis', '*.png'))

    def __len__(self):
        return len(self.infos)

    def __getitem__(self, index):
        image_name = self.infos[index].strip().split('/')[-1].split('\\')[-1].split('.')[0]

        vis = Image.open(os.path.join(self.root, 'pad', 'vis', image_name + '.png'))
        label = Image.open(os.path.join(self.root, 'pad', 'label', image_name + '.png'))
        ir = Image.open(os.path.join(self.root, 'pad', 'ir', image_name + '.png'))
        ir = ir.convert('RGB')
        ir_affine_deform = Image.open(os.path.join(self.root, 'pad', 'ir_affine_deform', image_name + '.png'))
        ir_affine_deform = ir_affine_deform.convert('RGB')

        sample = {
            'vis': vis,
            'ir': ir,
            'ir_affine_deform': ir_affine_deform,
            'label': label
        }

        if self.mode == 'train' and self.do_aug:
            sample = self.aug(sample)

        sample['vis'] = self.im_to_tensor(sample['vi'])  # [b, c, h, w]
        sample['ir'] = self.the_to_tensor(sample['ir'])  # [b, c, h, w]
        sample['ir_affine_deform'] = self.the_to_tensor(sample['ir_affine_deform'])  # [b, c, h, w]
        sample['label'] = torch.from_numpy(np.asarray(sample['label'], dtype=np.int64)).long()  # [b, h, w]

        return image_name, sample

    @property
    def cmap(self):
        return [
            (0, 0, 0),          # 0  Background
            (70, 70, 70),       # 1  Road
            (100, 40, 40),      # 2  Sidewalk
            (55, 90, 80),       # 3  Building
            (220, 20, 60),      # 4  Traffic Light
            (153, 153, 153),    # 5  Traffic Sign
            (157, 234, 50),     # 6  Vegetation
            (128, 64, 128),     # 7  Sky
            (245, 35, 232),     # 8  Person
            (255, 255, 0),      # 9  Car
            (102, 102, 156),    # 10 Truck
            (107, 142, 35),     # 11 Bus
            (70, 130, 180),     # 12 Motorcycle
            (81, 0, 81),        # 13 Bicycle
            (150, 100, 100)     # 14 Pole
        ]

