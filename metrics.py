#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time: 2025/3/18 9:46
# @Author: Tingyu Shi
# @File: metrics.py
# @Description: metrics

"""
# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

# Adapted from score written by meetshah1995
# https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/metrics.py

MIT License

Copyright (c) 2017 Meet Pragnesh Shah

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Import from https://github.com/YingLv1106/CAINet/blob/main/toolbox/metrics.py
"""
import numpy as np

import torch
from torch import nn

class runningScore(object):
    """
        n_classes: 数据集的类别，包括背景
        ignore_index: 需要忽略的类别id，一般为未标注id，e.g., CamVid.id_unlabel
    """

    def __init__(self, n_classes, ignore_index=None):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

        if ignore_index is None:
            self.ignore_index = None
        elif isinstance(ignore_index, int):
            self.ignore_index = (ignore_index,)
        else:
            try:
                self.ignore_index = tuple(ignore_index)
            except TypeError:
                raise ValueError("'ignore_index' must be an int or iterable")

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - Mean Acc: Mean Accuracy
            - Mean IoU: Mean Intersection over Union
            - Overall Acc: Overall Accuracy
            - Frequency Weighted IoU: Frequency Weighted Intersection over Union
            - cls_acc: Accuracy each Class
            - cls_iou: Intersection over Union each Class
        """

        hist = self.confusion_matrix

        # ignore unlabel
        if self.ignore_index is not None:
            for index in self.ignore_index:
                hist = np.delete(hist, index, axis=0)
                hist = np.delete(hist, index, axis=1)

        pixel_acc = np.diag(hist).sum() / hist.sum()
        acc = np.diag(hist) / hist.sum(axis=1)
        mean_acc = np.nanmean(acc)
        iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iou = np.nanmean(iou)
        freq = hist.sum(axis=1) / (hist.sum())
        fw_iou = (freq[freq > 0] * iou[freq > 0]).sum()

        # set unlabel as nan
        if self.ignore_index is not None:
            for index in self.ignore_index:
                iou = np.insert(iou, index, np.nan)

        cls_acc = dict(zip(range(self.n_classes), acc))
        cls_iou = dict(zip(range(self.n_classes), iou))

        return (
            {
                "Mean Acc: ": mean_acc,
                "Mean IoU: ": mean_iou,
                "Pixel Acc: ": pixel_acc,
                "Frequency Weighted IoU: ": fw_iou
            },
            cls_acc,
            cls_iou,
        )

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class averageMeter(object):
    """Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class NCC(nn.Module):
    """
        Normalized cross correlation.
    """
    def __init__(self):
        super(NCC, self).__init__()

    def similarity_loss(self, tgt, warped_img):
        sizes = np.prod(list(tgt.shape)[1:])
        flatten1 = torch.reshape(tgt, (-1, sizes))
        flatten2 = torch.reshape(warped_img, (-1, sizes))

        mean1 = torch.reshape(torch.mean(flatten1, dim=-1), (-1, 1))
        mean2 = torch.reshape(torch.mean(flatten2, dim=-1), (-1, 1))
        var1 = torch.mean((flatten1 - mean1) ** 2, dim=-1)
        var2 = torch.mean((flatten2 - mean2) ** 2, dim=-1)
        cov12 = torch.mean((flatten1 - mean1) * (flatten2 - mean2), dim=-1)
        pearson_r = cov12 / torch.sqrt((var1 + 1e-5) * (var2 + 1e-6))
        # ncc_value = torch.sum(1 - pearson_r)
        ncc_value = torch.sum(pearson_r)
        return ncc_value

    def forward(self, y_true, y_pred):
        return self.similarity_loss(y_true, y_pred)

class MSE(nn.Module):
    """
    Mean squared error loss.
    """
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)

class PCK(nn.Module):
    """
    Mean squared error loss.
    """
    def __init__(self):
        super(PCK, self).__init__()

    def forward(self, gt_flow, estimated_flow, threshold=5.):
        """
            Calculate the Percentage of Correct Keypoints (PCK) given a threshold.

            :param gt_flow: numpy array of ground truth optical flow.
            :param estimated_flow: numpy array of estimated optical flow.
            :param threshold: distance threshold in pixels to consider a keypoint correct.
            :return: PCK value.
        """
        threshold /= 255.

        # 计算预测和真实之间的差异
        flow_diff = torch.norm(estimated_flow - gt_flow, dim=0)
        # 计算满足阈值条件的正确关键点数量
        correct_keypoints = torch.sum(flow_diff < threshold)
        # 计算总关键点数量
        total_keypoints = estimated_flow.shape[1] * estimated_flow.shape[2]
        # 计算 PCK
        pck = (correct_keypoints / total_keypoints) * 100
        return pck

class MI(nn.Module):
    """
    PyTorch Module which calculates the Mutual Information (https://arxiv.org/abs/1910.12037).
    """

    def __init__(self):
        super(MI, self).__init__()

    def forward(self, a, b):
        # Calculate MI of a and b in the region of the overlap part

        # Calculate overlap region
        Ma, Na = a.shape
        Mb, Nb = b.shape
        M = min(Ma, Mb)
        N = min(Na, Nb)

        # Extract the overlapping regions
        a = a[:M, :N]
        b = b[:M, :N]

        # Normalize a and b
        if a.max() != a.min():
            a = ((a - a.min()) / (a.max() - a.min()) * 255).to(torch.uint8) + 1
        else:
            a = torch.zeros((M, N), dtype=torch.uint8, device=a.device) + 1

        if b.max() != b.min():
            b = ((b - b.min()) / (b.max() - b.min()) * 255).to(torch.uint8) + 1
        else:
            b = torch.zeros((M, N), dtype=torch.uint8, device=b.device) + 1

        # Flatten the tensors for vectorized processing
        a_flat = a.view(-1).long()
        b_flat = b.view(-1).long()

        # Initialize histograms using torch.zeros
        hab = torch.zeros((256, 256), device=a.device)
        ha = torch.zeros(256, device=a.device)
        hb = torch.zeros(256, device=b.device)

        # Use scatter_add_ to efficiently compute joint histogram
        hab.index_add_(0, a_flat, torch.eye(256, device=a.device)[b_flat])
        ha.index_add_(0, a_flat, torch.ones_like(a_flat, dtype=torch.float32))
        hb.index_add_(0, b_flat, torch.ones_like(b_flat, dtype=torch.float32))

        # Normalize histograms
        hab_sum = hab.sum()
        p_ab = hab / hab_sum
        p_ab = p_ab[p_ab > 0]  # Only non-zero probabilities

        # Calculate joint entropy
        Hab = -torch.sum(p_ab * torch.log(p_ab))

        # Normalize and compute entropy of a
        ha_sum = ha.sum()
        p_a = ha / ha_sum
        p_a = p_a[p_a > 0]  # Only non-zero probabilities
        Ha = -torch.sum(p_a * torch.log(p_a))

        # Normalize and compute entropy of b
        hb_sum = hb.sum()
        p_b = hb / hb_sum
        p_b = p_b[p_b > 0]  # Only non-zero probabilities
        Hb = -torch.sum(p_b * torch.log(p_b))

        # Calculate mutual information of a and b
        mi = Ha + Hb - Hab

        return mi


def reg_metric(fixed_images, moving_images, device):
    """
        Calculate MSE, NCC, MI and PCK-5 for each image in the batch.

        :param fixed_images: [B, C, H, W]
        :param moving_images: [B, C, H, W]
        :param device: CPU/CUDA
        :return: MSE, NCC, MI and PCK-5 for each image in the batch
    """
    mse_metric = MSE().to(device)
    ncc_metric = NCC().to(device)
    mi_metric = MI().to(device)
    pck_metric = PCK().to(device)

    batch_size = fixed_images.size(0)
    mse_values = []
    ncc_values = []
    mi_values = []
    pck_values = []

    for i in range(batch_size):
        fixed_image = fixed_images[i:i+1]
        moving_image = moving_images[i:i+1]

        mse = mse_metric(fixed_image, moving_image)
        ncc = ncc_metric(fixed_image, moving_image)
        mi = mi_metric(fixed_image.squeeze(), moving_image.squeeze())
        pck = pck_metric(fixed_image.squeeze(0), moving_image.squeeze(0), 5)

        mse_values.append(mse.item())
        ncc_values.append(ncc.item())
        mi_values.append(mi.item())
        pck_values.append(pck.item())

    return mse_values, ncc_values, mi_values, pck_values