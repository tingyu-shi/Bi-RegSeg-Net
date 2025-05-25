#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time: 2025/3/18 0:29
# @Author: Tingyu Shi
# @File: test_Seg.py
# @Description: test_Seg

import argparse
import json
import os
import argparse
import shutil
import time

import cv2
import torch
from torch.utils.data import DataLoader

from metrics import averageMeter, runningScore
from nets.model import Reg, Seg
from utils import get_logger, get_test_dataset, load_ckpt, class_to_RGB


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
    logdir = f'logs/{time.strftime("%Y-%m-%d-%H-%M-%S")}({cfg["dataset"]}-test-Seg)'
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
    model = Seg(cfg)
    model = load_ckpt(cfg['load_ckpt'], model, cfg['gpu'])
    model.to(device)

    # Define Meters
    test_running_metrics = runningScore(cfg['n_classes'])
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
            label = sample['label'].to(device)

            # Metric time
            time_start = time.time()
            out = model(ir, vis)
            time_meter.update(time.time() - time_start, cfg['batch_size'])

            # Metric Acc and IoU
            predict = out.max(1)[1].cpu().numpy()  # [b, h, w]
            label = label.cpu().numpy()
            test_running_metrics.update(label, predict)

            # Save predict images
            predict_RGB = class_to_RGB(predict[0], cfg['n_classes'], test_set.cmap)
            cv2.imwrite(os.path.join(cfg['save_image'], 'predict', image_name + '.png'), predict[0])
            cv2.imwrite(os.path.join(cfg['save_image'], 'predict_rgb', image_name + '.png'), predict_RGB[:, :, ::-1])

        # Update mIoU mAcc
        mIoU = test_running_metrics.get_scores()[0]['Mean IoU: ']
        mAcc = test_running_metrics.get_scores()[0]['Mean Acc: ']

        # Save logs
        logger.info(
            f'test average seg metric: mIoU={mIoU:.4f} mAcc={mAcc:.4f}, '
            f'inference time per image: {time_meter.avg:.4f}s, '
            f'inference frames per second: {(1 / time_meter.avg):.4f}fps'
        )

        metrics = test_running_metrics.get_scores()
        logger.info('Class Acc:')
        for k, v in metrics[1].items():
            logger.info(str(k) + '  ' + str(v))
        logger.info('Class IoU:')
        for k, v in metrics[2].items():
            logger.info(str(k) + '  ' + str(v))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="test")
    parser.add_argument("--config", default="./configs/test_mfnet.json", type=str, help="Configuration file to train")
    # parser.add_argument("--config", default="./configs/test_fmb.json", type=str, help="Configuration file to train")
    args = parser.parse_args()
    test(args)