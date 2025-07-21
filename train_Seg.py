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
from nets.model import Seg
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
    logdir = f'logs/{time.strftime("%Y-%m-%d-%H-%M-%S")}({cfg["dataset"]}-train-Seg)'
    os.makedirs(logdir, exist_ok=True)
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info(f'Conf | use logdir {logdir}')
    logger.info(f'Conf | use backbone {cfg["backbone"]}')

    # Config wandb
    # os.environ['WANDB_DISABLED'] = 'true'  # Disable wandb
    # os.environ["WANDB_MODE"] = "offline"  # Set wandb offline (default: online)
    miya = wandb.init(project='train_Seg', config=cfg, resume='allow')
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
    model = Seg(cfg)
    model.to(device)

    # Define optimizer
    seg_optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr_start_seg'], betas=(0.9, 0.999), weight_decay=cfg['weight_decay'])

    # Define lr scheduling
    scheduler = torch.optim.lr_scheduler.ExponentialLR(seg_optimizer, gamma=cfg['lr_decay'])

    # Define loss
    criterion_seg_1 = LovaszSoftmax().to(device)
    criterion_seg_2 = SoftCrossEntropyLoss(smooth_factor=0.1).to(device)

    # Define Meters
    train_loss_meter_seg = averageMeter()
    val_loss_meter_seg = averageMeter()
    train_loss_meter_seg_1 = averageMeter()
    val_loss_meter_seg_1 = averageMeter()
    train_loss_meter_seg_2 = averageMeter()
    val_loss_meter_seg_2 = averageMeter()
    val_running_metrics = runningScore(cfg['n_classes'])
    val_mse_meter = averageMeter()
    val_ncc_meter = averageMeter()
    val_mi_meter = averageMeter()
    val_pck_meter = averageMeter()

    # Define max_mIoU and max_mAcc
    max_mIoU = 0
    max_mAcc = 0

    for ep in range(cfg['epochs']):
        # Train
        model.train()
        for i, sample in enumerate(train_loader):
            sample = sample[1]
            vis = sample['vis'].to(device)
            ir = sample['ir'].to(device)
            label = sample['label'].to(device)
            out = model(ir, vis)

            # Update loss
            seg_loss_1 = criterion_seg_1(out, label)
            seg_loss_2 = criterion_seg_2(out, label)
            seg_loss = 0.5 * seg_loss_1 + 0.5 * seg_loss_2
            train_loss_meter_seg_1.update(seg_loss_1.item())
            train_loss_meter_seg_2.update(seg_loss_2.item())
            train_loss_meter_seg.update(seg_loss.item())

            seg_optimizer.zero_grad()
            seg_loss.backward()
            seg_optimizer.step()


        # Update lr
        lr_seg = seg_optimizer.param_groups[0]['lr']
        scheduler.step()

        # Val
        with torch.no_grad():
            model.eval()
            for i, sample in enumerate(val_loader):
                sample = sample[1]
                vis = sample['vis'].to(device)
                ir = sample['ir'].to(device)
                label = sample['label'].to(device)
                out = model(ir, vis)

                # Update loss
                seg_loss_1 = criterion_seg_1(out, label)
                seg_loss_2 = criterion_seg_2(out, label)
                seg_loss = 0.5 * seg_loss_1 + 0.5 * seg_loss_2
                val_loss_meter_seg_1.update(seg_loss_1.item())
                val_loss_meter_seg_2.update(seg_loss_2.item())
                val_loss_meter_seg.update(seg_loss.item())

                # Metric Acc and IoU
                predict = out.max(1)[1].cpu().numpy()  # [b, h, w]
                label = label.cpu().numpy()
                val_running_metrics.update(label, predict)

        # Update mIoU mAcc
        mIoU = val_running_metrics.get_scores()[0]['Mean IoU: ']
        mAcc = val_running_metrics.get_scores()[0]['Mean Acc: ']

        # Update MSE, NCC, MI and PCK
        MSE = val_mse_meter.avg
        NCC = val_ncc_meter.avg
        MI = val_mi_meter.avg
        PCK = val_pck_meter.avg

        # Save logs
        logger.info(
            f'Epoch | [{ep + 1:3d}/{cfg["epochs"]}] lr_seg={lr_seg:.12f}, '
            f'train/val loss={train_loss_meter_seg.avg:.5f}/{val_loss_meter_seg.avg:.5f}, '
            f'train/val seg_loss_SoftCrossEntropy={train_loss_meter_seg_1.avg:.5f}/{val_loss_meter_seg_1.avg:.5f}, '
            f'train/val seg_loss_LovaszSoftmax={train_loss_meter_seg_2.avg:.5f}/{val_loss_meter_seg_2.avg:.5f}, '
            f'val MSE={MSE:.4f} NCC={NCC:.4f} MI={MI:.4f} PCK-5={PCK:.4f}, '
            f'val mIoU={mIoU:.4f} mAcc={mAcc:.4f} '
        )

        # Save wandb
        miya.log({
            'epoch': ep + 1,
            'learning rate': lr_seg,
            'train loss': train_loss_meter_seg.avg,
            'val loss': val_loss_meter_seg.avg,
            'train seg_loss_SoftCrossEntropy': train_loss_meter_seg_1.avg,
            'val seg_loss_SoftCrossEntropy': val_loss_meter_seg_1.avg,
            'train seg_loss_LovaszSoftmax': train_loss_meter_seg_2.avg,
            'val seg_loss_LovaszSoftmax': val_loss_meter_seg_2.avg,
            'val MSE': MSE,
            'val NCC': NCC,
            'val MI': MI,
            'val PCK': PCK,
            'val mIoU': mIoU,
            'val mAcc': mAcc
        })

        # Save checkpoints
        os.makedirs(cfg['save_ckpt'], exist_ok=True)

        if mIoU > max_mIoU:
            max_mIoU = mIoU
            max_mAcc = mAcc
            logger.info(f'Save | save checkpoint {ep + 1}')
            save_ckpt(cfg['save_ckpt'], logdir.split('/')[-1] + '_epoch_' + f'{ep + 1:03d}' + '_MSE_' + f'{MSE:.4f}' + '_NCC_' + f'{NCC:.4f}' + '_MI_' + f'{MI:.4f}' + '_PCK_' + f'{PCK:.4f}' + '_mIoU_' + f'{mIoU:.4f}' + '_mAcc_' + f'{mAcc:.4f}', model)

        # Reset meters
        train_loss_meter_seg.reset()
        val_loss_meter_seg.reset()
        train_loss_meter_seg_1.reset()
        val_loss_meter_seg_1.reset()
        train_loss_meter_seg_2.reset()
        val_loss_meter_seg_2.reset()
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