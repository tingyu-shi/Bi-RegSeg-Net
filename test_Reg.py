#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time: 2025/3/18 0:29
# @Author: Tingyu Shi
# @File: test_Reg.py
# @Description: test_Reg

import argparse
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
from utils import get_logger, get_test_dataset, load_ckpt, imsave, save_flow


def test(args):
    """
        训练函数
        :param args: 训练配置
        :return: None
    """
    # torch.cuda.set_per_process_memory_fraction(1.0, 0)
    # Loading configs
    cfg = None
    with open(args.config, 'r') as fp:
        cfg = json.load(fp)
    assert cfg is not None
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
    test_running_metrics = runningScore(cfg['n_classes'])
    test_mse_meter = averageMeter()
    test_ncc_meter = averageMeter()
    test_mi_meter = averageMeter()
    test_pck_meter = averageMeter()
    time_meter = averageMeter()

    # Make Dirs
    os.makedirs(os.path.join(cfg['save_image'], 'reg_flow_disp'), exist_ok=True)
    os.makedirs(os.path.join(cfg['save_image'], 'reg_flow_image'), exist_ok=True)
    os.makedirs(os.path.join(cfg['save_image'], 'ir_affine_deform_reg'), exist_ok=True)
    # os.makedirs(os.path.join(cfg['save_image'], 'predict'), exist_ok=True)
    # os.makedirs(os.path.join(cfg['save_image'], 'predict_rgb'), exist_ok=True)

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
            # out = model(rgb, modal_x)

            time_meter.update(time.time() - time_start, cfg['batch_size'])

            # # Metric Acc and IoU
            # predict = out.max(1)[1].cpu().numpy()  # [b, h, w]
            # label = label.cpu().numpy()
            # test_running_metrics.update(label, predict)

            # Metric MSE, NCC, MI and PCK
            ir = ir[:, 0:1, :, :]
            warped_moving_ir = warped_moving_ir[:, 0:1, :, :]
            warped_moving_ir = (torch.round(warped_moving_ir * 255.).type(torch.FloatTensor) / 255.).to(device)
            mse_values, ncc_values, mi_values, pck_values = reg_metric(ir, warped_moving_ir, device)
            test_mse_meter.update(np.mean(mse_values), n=cfg['batch_size'])
            test_ncc_meter.update(np.mean(ncc_values), n=cfg['batch_size'])
            test_mi_meter.update(np.mean(mi_values), n=cfg['batch_size'])
            test_pck_meter.update(np.mean(pck_values), n=cfg['batch_size'])

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



        # # Update mIoU mAcc
        # mIoU = test_running_metrics.get_scores()[0]['Mean IoU: ']
        # mAcc = test_running_metrics.get_scores()[0]['Mean Acc: ']

        # Save logs
        logger.info(
            f'test MSE={test_mse_meter.avg:.4f} NCC={test_ncc_meter.avg:.4f} MI={test_mi_meter.avg:.4f} PCK-5={test_pck_meter.avg:.4f}%, '
            # f'test mIoU={mIoU:.4f} mAcc={mAcc:.4f} '
        )

        print('=' * 10, 'out', '=' * 10)
        print('inference time per image: %.4f'%time_meter.avg)
        print('inference fps: %.4f'%(1/time_meter.avg))
        # metrics = test_running_metrics.get_scores()
        # for k, v in metrics[0].items():
        #     print(k, v)
        # print('Class Acc:')
        # for k, v in metrics[1].items():
        #     print(k, v)
        # print('Class IoU:')
        # for k, v in metrics[2].items():
        #     print(k, v)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="test")
    parser.add_argument("--config", default="./configs/test_mfnet.json", type=str, help="Configuration file to train")
    # parser.add_argument("--config", default="./configs/test_fmb.json", type=str, help="Configuration file to train")
    args = parser.parse_args()
    test(args)