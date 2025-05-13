#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time: 2024/12/11 16:42
# @Author: Tingyu Shi
# @File: mfnet_in_memory.py
# @Description: mfnet_in_memory
import os

import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
from torchvision import transforms
from data.augmentations import Compose, ColorJitter, RandomCrop, RandomScale, RandomFlip


class MFNet_Dataset_InMemory(data.Dataset):
    def __init__(self, cfg, mode='test', do_aug=False):
        """
        一次性加载整个数据集到内存中。
        """
        self.name = cfg['dataset']
        self.mode = mode
        assert self.mode in ['train', 'val', 'test', 'test_day', 'test_night', 'black_list'], f'{self.mode} not support.'
        self.do_aug = do_aug
        self.mean = cfg['mean']
        self.std = cfg['std']

        ## pre-processing
        # Transformations for image data
        if cfg['normalize']:
            self.im_to_tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
                # transforms.Normalize([0.23584902, 0.26904956, 0.23294316], [0.23274839, 0.23397096, 0.23796114])
            ])
            # Transformations for thermal data
            self.the_to_tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
                # transforms.Normalize([0.34428293, 0.34428293, 0.34428293], [0.16110095, 0.16110095, 0.16110095])
            ])
        else:
            self.im_to_tensor = transforms.Compose([
                transforms.ToTensor()
            ])
            self.the_to_tensor = transforms.Compose([
                transforms.ToTensor()
            ])

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
                RandomCrop(crop_rate=cfg['crop_rate'], prob=cfg['crop_prob']),
            ])

        # 加载文件名
        with open(os.path.join(self.root, f'{self.mode}.txt'), 'r') as f:
            self.infos = [line.strip() for line in f.readlines() if not line.endswith('_flip\n')]

        # 预加载数据到内存
        self.data_in_memory = []
        for info in self.infos:
            image_name = info.strip()
            vis = Image.open(os.path.join(self.root, 'pad', 'vis', image_name + '.png'))
            ir_pseudo = Image.open(os.path.join(self.root, 'pad', 'ir_pseudo_stylebooth', image_name + '.png'))
            ir_pseudo = ir_pseudo.convert('RGB')
            label = Image.open(os.path.join(self.root, 'labels', image_name + '.png'))
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