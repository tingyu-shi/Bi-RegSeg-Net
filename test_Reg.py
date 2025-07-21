#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time: 2025/3/18 0:29
# @Author: Tingyu Shi
# @File: test_Reg.py
# @Description: test_Reg


import json
import os
import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from metrics import averageMeter, runningScore, reg_metric
from nets.model import Reg
from utils import get_logger, get_test_dataset, load_ckpt, imsave, save_flow, UnNormalize


def test(args):
    """
        训练函数
        :param args: 训练配置
        :return: None
    """
    # torch.cuda.set_per_process_memory_fraction(1.0, 0)
    # Loading configs
    with open(args.config, 'r') as fp:
        cfg = json.load(fp)
    cfg['batch_size'] = 1

    # Config logs
    logdir = f'logs/{time.strftime("%Y-%m-%d-%H-%M-%S")}({cfg["dataset"]}-test-Reg)'
    os.makedirs(logdir, exist_ok=True)
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info(f'Conf | use logdir {logdir}')
    logger.info(f'Conf | use backbone {cfg["backbone"]}')

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
    test_set = get_test_dataset(cfg)
    test_loader = DataLoader(test_set, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'], pin_memory=True)

    # Prepare the model
    model = Reg(cfg)
    model = load_ckpt(cfg['load_ckpt'], model, cfg['gpu'])
    model.to(device)

    # Define Meters
    test_mse_meter = averageMeter()
    test_ncc_meter = averageMeter()
    test_mi_meter = averageMeter()
    test_pck_meter = averageMeter()
    time_meter = averageMeter()

    # Make Dirs
    os.makedirs(os.path.join(cfg['save_image'], 'reg_flow_disp'), exist_ok=True)
    os.makedirs(os.path.join(cfg['save_image'], 'reg_flow_image'), exist_ok=True)

    # Test
    with torch.no_grad():
        model.eval()
        for i, sample in enumerate(test_loader):
            image_name = sample[0][0]
            sample = sample[1]
            vis = sample['vis'].to(device)
            ir = sample['ir'].to(device)
            ir_affine_deform = sample['ir_affine_deform'].to(device)
            ir_pseudo = sample['ir_pseudo'].to(device)
            label = sample['label'].to(device)

            # Metric time
            time_start = time.time()
            warped_moving_ir, rewarped_fixed_ir_pseudo, flow = model(ir_affine_deform, ir_pseudo, vis)
            time_meter.update(time.time() - time_start, cfg['batch_size'])

            # Save registrated ir_affine_deform
            warped_moving_ir_path = Path(cfg['save_image']) / 'ir_affine_deform_reg'
            imsave(warped_moving_ir, warped_moving_ir_path, image_name + '.png')

            # Save estimated flow
            flow_disp_npy = flow.data.cpu().numpy()
            flow_disp_path = Path(cfg['save_image'], 'reg_flow_disp', image_name + '.npy')
            np.save(flow_disp_path, flow_disp_npy)

            # Save flow images
            flow_path = Path(cfg['save_image']) / 'reg_flow_image'
            save_flow(flow, flow_path, image_name + '.png')

            # Save logs
            logger.info(
                image_name+f' -- MSE={test_mse_meter.val:.4f} -- NCC={test_ncc_meter.val:.4f} -- MI={test_mi_meter.val:.4f} -- PCK-5={test_pck_meter.val:.4f}%'
            )

        # Save logs
        logger.info(
            f'test average reg metric: MSE={test_mse_meter.avg:.4f} NCC={test_ncc_meter.avg:.4f} MI={test_mi_meter.avg:.4f} PCK-5={test_pck_meter.avg:.4f}%, '
            f'inference time per image: {time_meter.avg:.4f}s, '
            f'inference frames per second: {(1/time_meter.avg):.4f}fps'
        )

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="test")
    parser.add_argument("--config", default="./configs/test_mfnet.json", type=str, help="Configuration file to train")
    # parser.add_argument("--config", default="./configs/test_fmb.json", type=str, help="Configuration file to train")
    args = parser.parse_args()
    test(args)