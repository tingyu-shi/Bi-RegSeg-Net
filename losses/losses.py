#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time: 2024/9/24 18:56
# @Author: Tingyu Shi
# @File: losses.py
# @Description: losses.py
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision

from losses.lovasz_losses import lovasz_softmax


class LossFunction_Dense(nn.Module):
    def __init__(self):
        super(LossFunction_Dense, self).__init__()
        self.gradient_loss = gradient_loss()
        self.feat_loss = VGGLoss()

    def forward(self, y, y_f, tgt, src, flow): # tgt: torch.Size([16, 1, 224, 224])
        '''
        配准常用的loss函数
        :param y: warped_image = moving_image * flow = moving_image施加flow场
        :param y_f: warped_image_f = fixed_image * -flow = fixed_image施加逆向flow场
        :param tgt: fixed_image
        :param src: moving_image
        :param flow: flow场/光流场
        :param flow1: 废弃
        :param flow2: 废弃
        :return:
        '''
        hyper_ncc = 1
        hyper_grad = 10
        hyper_feat = 1

        # similarity loss
        ncc_1 = torch.nn.functional.l1_loss(tgt, y)
        ncc_2 = torch.nn.functional.l1_loss(src, y_f)
        ncc = ncc_1 + 0.2*ncc_2

        # feature loss
        feat_1 = self.feat_loss(y, tgt)
        feat_2 = self.feat_loss(y_f, src)
        feat = feat_1 + 0.2*feat_2

        # gradient loss
        grad = self.gradient_loss(flow)

        # total loss
        loss = hyper_feat * feat + hyper_grad * grad
        return loss, feat, ncc, grad

class gradient_loss(nn.Module):
    def __init__(self):
        super(gradient_loss, self).__init__()

    def forward(self, s, penalty='l2'):
        dy = torch.abs(s[:, :, 1:, :] - s[:, :, :-1, :])
        dx = torch.abs(s[:, :, :, 1:] - s[:, :, :, :-1])
        if (penalty == 'l2'):
            dy = dy * dy
            dx = dx * dx
        d = torch.mean(dx) + torch.mean(dy)
        return d / 2.0

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X) # torch.Size([1, 64, 256, 256])
        h_relu2 = self.slice2(h_relu1) # torch.Size([1, 128, 128, 128])
        h_relu3 = self.slice3(h_relu2) # torch.Size([1, 256, 64, 64])
        h_relu4 = self.slice4(h_relu3) # torch.Size([1, 512, 32, 32])
        h_relu5 = self.slice5(h_relu4) # torch.Size([1, 512, 16, 16])
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19()
        if torch.cuda.is_available():
            self.vgg.cuda()
        self.vgg.eval()
        set_requires_grad(self.vgg, False)
        self.L1Loss = nn.L1Loss()
        self.criterion2 = nn.MSELoss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 , 1.0]

    def forward(self, x, y):
        contentloss = 0
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)
        x_vgg = self.vgg(x)
        with torch.no_grad():
            y_vgg = self.vgg(y)

        contentloss += self.L1Loss(x_vgg[3], y_vgg[3].detach())

        return contentloss

class LovaszSoftmax(nn.Module):
    def __init__(self, classes='present', per_image=False, ignore_index=255):
        super(LovaszSoftmax, self).__init__()
        self.smooth = classes
        self.per_image = per_image
        self.ignore_index = ignore_index
    def forward(self, output, target):
        logits = F.softmax(output, dim=1)
        loss = lovasz_softmax(probas=logits, labels=target, ignore=self.ignore_index)
        return loss