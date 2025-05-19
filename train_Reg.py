#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time: 2025/3/18 0:28
# @Author: Tingyu Shi
# @File: train_Reg.py
# @Description: train_Reg

import json
import os
import argparse
import shutil
import time

import numpy as np
import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader

from losses.losses import LovaszSoftmax, LossFunction_Dense
from losses.soft_ce_loss import SoftCrossEntropyLoss
from metrics import averageMeter, runningScore, reg_metric
from nets.model import Reg
from utils import get_train_val_dataset, save_ckpt, get_logger, UnNormalize


def train(args):
    """
        训练函数
        :param args: 训练配置
        :return: None
    """
    # torch.cuda.set_per_process_memory_fraction(1.0, 0)
    # Loading configs
    with open(args.config, 'r') as fp:
        cfg = json.load(fp)

    # Config logs
    logdir = f'logs/{time.strftime("%Y-%m-%d-%H-%M-%S")}({cfg["dataset"]}-train-Reg)'
    os.makedirs(logdir, exist_ok=True)
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info(f'Conf | use logdir {logdir}')
    logger.info(f'Conf | use backbone {cfg["backbone"]}')

    # Config wandb
    miya = wandb.init(project='train_Reg', config=cfg, resume='allow')
    miya.config.update(dict(logdir=logdir))

    # Handling devices
    print('===> Handling devices')
    if cfg['gpu'] == '-1':
        device = torch.device('cpu')
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg['gpu']
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            raise Exception("No GPU found...")

    # Loading datasets
    print("===> Loading datasets")
    train_set, val_set = get_train_val_dataset(cfg)
    train_loader = DataLoader(train_set, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'], pin_memory=True)

    # Prepare the model
    model = Reg(cfg)
    model.to(device)

    # Define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr_start_reg'], betas=(0.9, 0.999), weight_decay=cfg['weight_decay'])

    # Define lr scheduling
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg['lr_decay'])

    # Define loss
    criterion_1 = LossFunction_Dense().to(device)
    criterion_2 = LovaszSoftmax().to(device)
    criterion_3 = nn.CrossEntropyLoss().to(device)
    criterion_3 = SoftCrossEntropyLoss(smooth_factor=0.1).to(device)

    # Define Meters
    # train_loss_meter_1 = averageMeter()
    # train_loss_meter_2 = averageMeter()
    # train_loss_meter_3 = averageMeter()
    # val_loss_meter_1 = averageMeter()
    # val_loss_meter_2 = averageMeter()
    # val_loss_meter_3 = averageMeter()
    train_loss_meter = averageMeter()
    val_loss_meter = averageMeter()
    val_running_metrics = runningScore(cfg['n_classes'])
    val_mse_meter = averageMeter()
    val_ncc_meter = averageMeter()
    val_mi_meter = averageMeter()
    val_pck_meter = averageMeter()

    # Define max_mIoU and max_mAcc
    max_mIoU = 0
    max_mAcc = 0

    # Define max_MSE, max_NCC, max_MI and max_PCK
    min_MSE = 1
    max_NCC = 0
    max_MI = 0
    max_PCK = 0

    # Define UnNormalize
    unnorm = UnNormalize(mean=cfg['mean'], std=cfg['std']).to(device)

    for ep in range(cfg['epochs']):
        # Train
        model.train()
        for i, sample in enumerate(train_loader):
            sample = sample[1]
            vis = sample['vis'].to(device)
            ir = sample['ir'].to(device)
            ir_pseudo = sample['ir_pseudo'].to(device)
            ir_affine_deform = sample['ir_affine_deform'].to(device)
            label = sample['label'].to(device)
            warped_moving_ir, rewarped_fixed_ir_pseudo, flow = model(ir_affine_deform, ir_pseudo, vis)

            # Update loss
            # loss_1 = criterion_1(out, label)
            # loss_2 = criterion_2(out, label)
            # seg_loss = 0.5 * loss_1 + 0.5 * loss_2
            # loss_3 = criterion_3(out, label)
            # seg_loss = 0.33 * loss_1 + 0.33 * loss_2 + 0.33 * loss_3
            # train_loss_meter_1.update(loss_1.item())
            # train_loss_meter_2.update(loss_2.item())
            # train_loss_meter_3.update(loss_3.item())
            reg_loss, _, _, _ = criterion_1(warped_moving_ir, rewarped_fixed_ir_pseudo, ir_pseudo, ir_affine_deform, flow)
            train_loss_meter.update(reg_loss.item())

            # Backpropagation and optimize
            optimizer.zero_grad()
            # seg_loss.backward()
            reg_loss.backward()
            optimizer.step()

        # Update lr
        lr = optimizer.param_groups[0]['lr']
        # scheduler.step()

        # Val
        with torch.no_grad():
            model.eval()
            for i, sample in enumerate(val_loader):
                sample = sample[1]
                vis = sample['vis'].to(device)
                ir = sample['ir'].to(device)
                ir_affine_deform = sample['ir_affine_deform'].to(device)
                ir_pseudo = sample['ir_pseudo'].to(device)
                label = sample['label'].to(device)
                warped_moving_ir, rewarped_fixed_ir_pseudo, flow = model(ir_affine_deform, ir_pseudo, vis)
                # out = model(rgb, modal_x)

                # Update loss
                # loss_1 = criterion_1(out, label)
                # loss_2 = criterion_2(out, label)
                # seg_loss = 0.5 * loss_1 + 0.5 * loss_2
                # loss_3 = criterion_3(out, label)
                # seg_loss = 0.33 * loss_1 + 0.33 * loss_2 + 0.33 * loss_3
                # val_loss_meter_1.update(loss_1.item())
                # val_loss_meter_2.update(loss_2.item())
                # val_loss_meter_3.update(loss_3.item())
                # seg_loss = criterion_1(out, label)
                reg_loss, _, _, _ = criterion_1(warped_moving_ir, rewarped_fixed_ir_pseudo, ir_pseudo, ir_affine_deform, flow)
                val_loss_meter.update(reg_loss.item())

                # # Metric Acc and IoU
                # predict = out.max(1)[1].cpu().numpy()  # [b, h, w]
                # label = label.cpu().numpy()
                # test_running_metrics.update(label, predict)

                # Metric MSE, NCC, MI and PCK
                if cfg['normalize']:
                    ir = unnorm(ir)
                    warped_moving_ir = unnorm(warped_moving_ir)
                ir = ir[:, 0:1, :, :]
                warped_moving_ir = warped_moving_ir[:, 0:1, :, :]
                warped_moving_ir = (torch.round(warped_moving_ir * 255.).type(torch.FloatTensor) / 255.).to(device)
                mse_values, ncc_values, mi_values, pck_values = reg_metric(ir, warped_moving_ir, device)
                val_mse_meter.update(np.mean(mse_values), n=cfg['batch_size'])
                val_ncc_meter.update(np.mean(ncc_values), n=cfg['batch_size'])
                val_mi_meter.update(np.mean(mi_values), n=cfg['batch_size'])
                val_pck_meter.update(np.mean(pck_values), n=cfg['batch_size'])

        # # Update mIoU mAcc
        # mIoU = val_running_metrics.get_scores()[0]['Mean IoU: ']
        # mAcc = val_running_metrics.get_scores()[0]['Mean Acc: ']

        # Update MSE, NCC, MI and PCK
        MSE = val_mse_meter.avg
        NCC = val_ncc_meter.avg
        MI = val_mi_meter.avg
        PCK = val_pck_meter.avg

        # Save logs
        logger.info(
            f'Epoch | [{ep + 1:3d}/{cfg["epochs"]}] lr={lr:.12f}, '
            f'train/val loss={train_loss_meter.avg:.5f}/{val_loss_meter.avg:.5f}, '
            # f'train/val loss_CrossEntropy={train_loss_meter_1.avg:.5f}/{val_loss_meter_1.avg:.5f}, '
            # f'train/val loss_SoftCrossEntropy={train_loss_meter_1.avg:.5f}/{val_loss_meter_1.avg:.5f}, '
            # f'train/val loss_LovaszSoftmax={train_loss_meter_2.avg:.5f}/{val_loss_meter_2.avg:.5f}, '
            # f'train/val loss_Dice={train_loss_meter_3.avg:.5f}/{val_loss_meter_3.avg:.5f}, '
            f'val MSE={MSE:.4f} NCC={NCC:.4f} MI={MI:.4f} PCK-5={PCK:.4f}%, '
            # f'val mIoU={mIoU:.4f} mAcc={mAcc:.4f} '
        )

        # Save wandb
        miya.log({
            'epoch': ep + 1,
            'learning rate': lr,
            'train loss': train_loss_meter.avg,
            'val loss': val_loss_meter.avg,
            # 'train loss (CrossEntropy)': train_loss_meter_1.avg,
            # 'val loss (CrossEntropy)': val_loss_meter_1.avg,
            # 'train loss (SoftCrossEntropy)': train_loss_meter_1.avg,
            # 'val loss (SoftCrossEntropy)': val_loss_meter_1.avg,
            # 'train loss (LovaszSoftmax)': train_loss_meter_2.avg,
            # 'val loss (LovaszSoftmax)': val_loss_meter_2.avg,
            # 'train loss (Dice)': train_loss_meter_3.avg,
            # 'val loss (Dice)': val_loss_meter_3.avg,
            'val MSE': MSE,
            'val NCC': NCC,
            'val MI': MI,
            'val PCK': PCK,
            # 'val mIoU': mIoU,
            # 'val mAcc': mAcc
        })

        # Save checkpoints
        os.makedirs(cfg['save_ckpt'], exist_ok=True)
        # if mIoU > max_mIoU and mAcc > max_mAcc:
        #     max_mIoU = mIoU
        #     max_mAcc = mAcc
        #     logger.info(f'Save | save checkpoint {ep + 1}')
        #     save_ckpt(cfg['save_ckpt'], logdir.split('/')[-1] + '_epoch_' + f'{ep + 1:03d}' + '_mIoU_' + f'{max_mIoU:.4f}' + '_mAcc_' + f'{max_mAcc:.4f}', model)
        # elif mIoU > max_mIoU:
        #     max_mIoU = mIoU
        #     logger.info(f'Save | save checkpoint {ep + 1}')
        #     save_ckpt(cfg['save_ckpt'], logdir.split('/')[-1] + '_epoch_' + f'{ep + 1:03d}' + '_mIoU_' + f'{max_mIoU:.4f}', model)
        # elif mAcc > max_mAcc:
        #     max_mAcc = mAcc
        #     logger.info(f'Save | save checkpoint {ep + 1}')
        #     save_ckpt(cfg['save_ckpt'], logdir.split('/')[-1] + '_epoch_' + f'{ep + 1:03d}' + '_mAcc_' + f'{max_mAcc:.4f}', model)

        comparison_results = [MSE <= min_MSE, NCC >= max_NCC, MI >= max_MI, PCK >= max_PCK]
        if sum(comparison_results) >= 2:
            min_MSE, max_NCC, max_MI, max_PCK = MSE, NCC, MI, PCK
            logger.info(f'Save | save checkpoint {ep + 1}')
            save_ckpt(cfg['save_ckpt'], logdir.split('/')[-1] + '_epoch_' + f'{ep + 1:03d}' + '_MSE_' + f'{min_MSE:.4f}' + '_NCC_' + f'{max_NCC:.4f}' + '_MI_' + f'{max_MI:.4f}' + '_PCK_' + f'{max_PCK:.4f}', model)

        # Reset meters
        # train_loss_meter_1.reset()
        # train_loss_meter_2.reset()
        # train_loss_meter_3.reset()
        # val_loss_meter_1.reset()
        # val_loss_meter_2.reset()
        # val_loss_meter_3.reset()
        train_loss_meter.reset()
        val_loss_meter.reset()
        val_running_metrics.reset()
        val_mse_meter.reset()
        val_ncc_meter.reset()
        val_mi_meter.reset()
        val_pck_meter.reset()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="train")
    parser.add_argument("--config", default="./configs/train_mfnet.json", type=str, help="Configuration file to train")
    # parser.add_argument("--config", default="./configs/train_fmb.json", type=str, help="Configuration file to train")
    args = parser.parse_args()
    train(args)
