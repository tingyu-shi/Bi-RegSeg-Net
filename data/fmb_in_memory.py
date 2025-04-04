#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time: 2024/12/11 21:53
# @Author: Tingyu Shi
# @File: fmb_in_memory.py
# @Description: fmb_in_memory
import glob
import os

import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
from torchvision import transforms
from data.augmentations import Compose, ColorJitter, RandomCrop, RandomScale, RandomFlip


class FMB_Dataset_InMemory(data.Dataset):
    def __init__(self, cfg, mode='test', do_aug=False):
        """
        一次性加载整个数据集到内存中。
        """
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

        self.root = os.path.join(cfg['root'], self.mode)
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
                RandomCrop(crop_rate=cfg['crop_rate'], prob=cfg['crop_prob']),
            ])

        # 加载文件名
        self.infos = glob.glob(os.path.join(self.root, 'Label', '*.png'))

        # 预加载数据到内存
        self.data_in_memory = []
        for info in self.infos:
            image_name = info.strip().split('/')[-1].split('\\')[-1].split('.')[0]

            vis = Image.open(os.path.join(self.root, 'pad', 'vis', image_name + '.png'))
            ir_pseudo = Image.open(os.path.join(self.root, 'pad', 'ir_pseudo_stylebooth', image_name + '.png'))
            ir_pseudo = ir_pseudo.convert('RGB')
            label = Image.open(os.path.join(self.root, 'ori', 'Label', image_name + '.png'))
            ir = Image.open(os.path.join(self.root, 'pad', 'ir', image_name + '.png'))
            ir = ir.convert('RGB')
            ir_affine_deform = Image.open(os.path.join(self.root, 'pad', 'ir_affine_deform', image_name + '.png'))
            ir_affine_deform = ir_affine_deform.convert('RGB')

            sample = {
                'vis': self.im_to_tensor(vis),  # [b, c, h, w]
                'ir_pseudo': self.im_to_tensor(ir_pseudo),  # [b, c, h, w]
                'ir': self.im_to_tensor(ir),  # [b, c, h, w]
                'ir_affine_deform': self.im_to_tensor(ir_affine_deform),  # [b, c, h, w]
                'label': torch.from_numpy(np.asarray(label, dtype=np.int64)).long()  # [b, h, w]
            }

            if self.mode == 'train' and self.do_aug:
                sample = self.aug(sample)

            self.data_in_memory.append((image_name, sample))

    def __len__(self):
        return len(self.data_in_memory)

    def __getitem__(self, index):
        """
        直接从内存中返回数据。
        """
        return self.data_in_memory[index]

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