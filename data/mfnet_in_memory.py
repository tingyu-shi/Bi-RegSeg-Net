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
        for image_name in self.infos:
            # 加载图像数据和标签
            im_the = Image.open(os.path.join(self.root, 'images', image_name + '.png'))
            im_the = np.asarray(im_the)
            rgb = Image.fromarray(im_the[:, :, :3])
            t = Image.fromarray(np.concatenate([im_the[:, :, 3:], im_the[:, :, 3:], im_the[:, :, 3:]], axis=2))
            label = Image.open(os.path.join(self.root, 'labels', image_name + '.png'))

            # 存储到内存
            sample = {
                'rgb': self.im_to_tensor(rgb),
                't': self.the_to_tensor(t),
                'label': torch.from_numpy(np.asarray(label, dtype=np.int64)).long(),
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