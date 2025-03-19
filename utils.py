#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time: 2025/3/18 9:48
# @Author: Tingyu Shi
# @File: utils.py
# @Description: utils
import logging
import os
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data.fmb import FMB_Dataset
from data.fmb_in_memory import FMB_Dataset_InMemory
from data.mfnet import MFNet_Dataset
from data.mfnet_in_memory import MFNet_Dataset_InMemory


class ClassWeight(object):

    def __init__(self, method):
        assert method in ['no', 'enet', 'median_freq_balancing']
        self.method = method

    def get_weight(self, data, num_classes):
        if self.method == 'no':
            return np.ones(num_classes)
        if self.method == 'enet':
            dataloader = DataLoader(data, batch_size=1, shuffle=False,pin_memory=True)
            return self._enet_weighing(dataloader, num_classes)
        if self.method == 'median_freq_balancing':
            dataloader = DataLoader(data, batch_size=1, shuffle=False, pin_memory=True)
            return self._median_freq_balancing(dataloader, num_classes)

    def _enet_weighing(self, dataloader, num_classes, c=1.02):
        """Computes class weights as described in the ENet paper:

            w_class = 1 / (ln(c + p_class)),

        where c is usually 1.02 and p_class is the propensity score of that
        class:

            propensity_score = freq_class / total_pixels.

        References: https://arxiv.org/abs/1606.02147

        Keyword arguments:
        - dataloader (``data.Dataloader``): A data loader to iterate over the
        dataset.
        - num_classes (``int``): The number of classes.
        - c (``int``, optional): AN additional hyper-parameter which restricts
        the interval of values for the weights. Default: 1.02.

        """
        print('computing class weight .......................')
        class_count = 0
        total = 0
        for i, label in tqdm(enumerate(dataloader), total=len(dataloader)):
            label = label.numpy()

            # Flatten label
            flat_label = label.flatten()

            # Sum up the number of pixels of each class and the total pixel
            # counts for each label
            class_count += np.bincount(flat_label, minlength=num_classes)
            total += flat_label.size

        # Compute propensity score and then the weights for each class
        propensity_score = class_count / total
        class_weights = 1 / (np.log(c + propensity_score))

        return class_weights

    def _median_freq_balancing(self, dataloader, num_classes):
        """Computes class weights using median frequency balancing as described
        in https://arxiv.org/abs/1411.4734:

            w_class = median_freq / freq_class,

        where freq_class is the number of pixels of a given class divided by
        the total number of pixels in images where that class is present, and
        median_freq is the median of freq_class.

        Keyword arguments:
        - dataloader (``data.Dataloader``): A data loader to iterate over the
        dataset.
        whose weights are going to be computed.
        - num_classes (``int``): The number of classes

        """
        print('computing class weight .......................')
        class_count = 0
        total = 0
        for i, label in tqdm(enumerate(dataloader), total=len(dataloader)):
            label = label.numpy()

            # Flatten label
            flat_label = label.flatten()

            # Sum up the class frequencies
            bincount = np.bincount(flat_label, minlength=num_classes)

            # Create of mask of classes that exist in the label
            mask = bincount > 0
            # Multiply the mask by the pixel count. The resulting array has
            # one element for each class. The value is either 0 (if the class
            # does not exist in the label) or equal to the pixel count (if
            # the class exists in the label)
            total += mask * flat_label.size

            # Sum up the number of pixels found for each class
            class_count += bincount

        # Compute the frequency and its median
        freq = class_count / total
        med = np.median(freq)

        return med / freq

def load_ckpt(path, model, gpu):
    """
    使用 load_ckpt 函数适合于检查点和模型结构可能存在差异的情况，提供了更高的灵活性和安全性。
    :param path: checkpoints路径
    :param model: model实例
    :return:
    """
    if gpu == '-1':
        _dict = torch.load(path, map_location='cpu')
    else:
        _dict = torch.load(path, map_location='cuda')
    model_dict = model.state_dict()
    _dict = {k: v for k, v in _dict.items() if k in model_dict}
    model_dict.update(_dict)
    model.load_state_dict(model_dict)
    return model


def color_map(N=256, normalized=False):
    """
        Return Color Map in PASCAL VOC format
    """

    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    dtype = "float32" if normalized else "uint8"
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255.0 if normalized else cmap
    return cmap


def class_to_RGB(label, N, cmap=None, normalized=False):
    """
        label: 2D numpy array with pixel-level classes shape=(h, w)
        N: number of classes, including background, should in [0, 255]
        cmap: list of colors for N class (include background) \
              if None, use VOC default color map.
        normalized: RGB in [0, 1] if True else [0, 255] if False

        :return: 上色好的3D RGB numpy array shape=(h, w, 3)
    """
    dtype = "float32" if normalized else "uint8"

    assert len(label.shape) == 2, f'label should be 2D, not {len(label.shape)}D'
    label_class = np.asarray(label)

    label_color = np.zeros((label.shape[0], label.shape[1], 3), dtype=dtype)

    if cmap is None:
        # 0表示背景为[0 0 0]黑色,1~N表示N个类别彩色
        cmap = color_map(N, normalized=normalized)
    else:
        cmap = np.asarray(cmap, dtype=dtype)
        cmap = cmap / 255.0 if normalized else cmap

    assert cmap.shape[0] == N, f'{N} classes and {cmap.shape[0]} colors not match.'

    # 给每个类别根据color_map上色
    for i_class in range(N):
        label_color[label_class == i_class] = cmap[i_class]

    return label_color


def class_to_binary(label, normalized=False):
    """
        label: 2D numpy array with pixel-level classes shape=(h, w)

        :return: 二值化的2D numpy array shape=(h, w)
    """
    assert len(label.shape) == 2, f'label should be 2D, not {len(label.shape)}D'
    label_binary = np.zeros_like(label, dtype="uint8")

    # 将非零像素设置为 255
    if normalized:
        label_binary[label != 0] = 1
    else:
        label_binary[label != 0] = 255

    return label_binary


def class_to_boundary(label, sigma=0.33):
    """
        label: 2D numpy array with pixel-level classes shape=(h, w)
        sigma: Standard deviation of the Gaussian kernel used in Canny edge detection

        :return: 边界图的2D numpy array shape=(h, w)
    """
    # Convert label to binary
    label_binary = class_to_binary(label)

    # Compute the median of the single channel pixel intensities
    v = np.median(label_binary)

    # Apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    # Generate the boundary image using Canny edge detection
    label_boundary = cv2.Canny(label_binary, lower, upper)

    return label_boundary

def class_to_attention(label, kernel_size1=3, kernel_size2=7, sigmaX=0): #:TODO 原本kernel_size=3 应该设成几？
    """
        label: 2D numpy array with pixel-level classes shape=(h, w)
        kernel_size: Size of the kernel for erosion and dilation operations
        sigmaX: Standard deviation in the X direction for Gaussian blur

        :return: 注意力图的2D numpy array shape=(h, w)
    """
    # Convert label to binary
    label_binary = class_to_binary(label)

    # Create a structuring element for erosion and dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size1, kernel_size1))

    # Apply corrosion
    label_eroded = cv2.erode(label_binary, kernel, iterations=1)

    # Apply expansion
    label_dilated = cv2.dilate(label_eroded, kernel, iterations=1)

    # Apply Gaussian fuzzy
    label_attention = cv2.GaussianBlur(label_dilated, (kernel_size2, kernel_size2), sigmaX)

    return label_attention


def save_ckpt(path, prefix, model):
    """
    保存模型的checkpoint到指定目录。

    该函数首先确定是否在多GPU环境下使用模型，如果是，则提取module的状态字典；
    否则直接提取模型的状态字典。然后将状态字典保存到指定的日志目录下，文件名为'model.pth'。

    参数:
    - logdir (str): 保存模型checkpoint的目录路径。
    - model (torch.nn.Module): 需要保存的模型实例。

    返回:
    无返回值。
    """
    # 根据模型是否有'module'属性来判断是否是DataParallel模型
    # 如果是，则提取module内的状态字典；否则直接提取模型状态字典
    state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()

    # 将模型状态字典保存到指定路径
    torch.save(state_dict, os.path.join(path, prefix + '.pth'))


def get_train_val_dataset(cfg):
    if cfg['in_memory']:
        if cfg['dataset'] == 'MFNet':
            return MFNet_Dataset_InMemory(cfg, mode='train', do_aug=True), MFNet_Dataset_InMemory(cfg, mode='test', do_aug=False)
        elif cfg['dataset'] == 'PST900':
            return PST900_Dataset_InMemory(cfg, mode='train', do_aug=True), PST900_Dataset_InMemory(cfg, mode='test', do_aug=False)
        elif cfg['dataset'] == 'FMB':
            return FMB_Dataset_InMemory(cfg, mode='train', do_aug=True), FMB_Dataset_InMemory(cfg, mode='test', do_aug=False)
        else:
            raise Exception("Unknown dataset...")
    else:
        if cfg['dataset'] == 'MFNet':
            return MFNet_Dataset(cfg, mode='train', do_aug=True), MFNet_Dataset(cfg, mode='test', do_aug=False)
        elif cfg['dataset'] == 'PST900':
            return PST900_Dataset(cfg, mode='train', do_aug=True), PST900_Dataset(cfg, mode='test', do_aug=False)
        elif cfg['dataset'] == 'FMB':
            return FMB_Dataset(cfg, mode='train', do_aug=True), FMB_Dataset(cfg, mode='test', do_aug=False)
        else:
            raise Exception("Unknown dataset...")

def get_test_dataset(cfg):
    if cfg['in_memory']:
        if cfg['dataset'] == 'MFNet':
            return MFNet_Dataset_InMemory(cfg, mode='test', do_aug=False)
        elif cfg['dataset'] == 'PST900':
            return PST900_Dataset_InMemory(cfg, mode='test', do_aug=False)
        elif cfg['dataset'] == 'FMB':
            return FMB_Dataset_InMemory(cfg, mode='test', do_aug=False)
        else:
            raise Exception("Unknown dataset...")
    else:
        if cfg['dataset'] == 'MFNet':
            return MFNet_Dataset(cfg, mode='test', do_aug=False)
        elif cfg['dataset'] == 'PST900':
            return PST900_Dataset(cfg, mode='test', do_aug=False)
        elif cfg['dataset'] == 'FMB':
            return FMB_Dataset(cfg, mode='test', do_aug=False)
        else:
            raise Exception("Unknown dataset...")


def get_mean_std(data):
    '''
    Compute mean and std for data
    :param data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    '''
    print('Compute mean and std for', data.name, 'Dataset')
    print(len(data))
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    data_loader = torch.utils.data.DataLoader(
        data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1
    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    return list(mean.numpy()), list(std.numpy())


def get_logger(logdir):

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    logname = logdir.split('/')[-1]
    log_file = os.path.join(logdir, logname + '.log')

    # create log
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)

    # Formatter 设置日志输出格式
    formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # StreamHandler 日志输出1 -> 到控制台
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # FileHandler 日志输出2 -> 保存到文件log_file
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger