import json
import os
import random

import cv2
import torch
import numpy as np
from PIL import Image
from matplotlib import cm
from torchvision.transforms import transforms

from nets.model import RegSeg, Reg
from utils import load_ckpt, class_to_RGB


cmap = [
        (0, 0, 0),          # 0  Background
        (179, 228, 228),    # 1  Road
        (181, 57, 133),     # 2  Sidewalk
        (67, 162, 177),     # 3  Building
        (200, 178, 50),     # 4  Traffic Light
        (132, 45, 199),     # 5  Traffic Sign
        (66, 172, 84),      # 6  Vegetation
        (179, 73, 79),      # 7  Sky
        (76, 99, 166),      # 8  Person
        (66, 121, 253),     # 9  Car
        (137, 165, 91),     # 10 Truck
        (155, 97, 152),     # 11 Bus
        (105, 153, 140),    # 12 Motorcycle
        (222, 215, 158),    # 13 Bicycle
        (135, 113, 90)      # 14 Pole
    ]


def random_num(size, end):
    range_ls = [i for i in range(end)]
    num_ls = []
    for i in range(size):
        num = random.choice(range_ls)
        range_ls.remove(num)
        num_ls.append(num)
    return num_ls


def save_selected_channels_png(fx1, warped_fx1, selected_channels, save_dir):
    """
    保存指定通道的特征图为彩色PNG图片，使用viridis颜色映射

    Args:
        fx1: 原始特征图 [B, C, H, W]
        warped_fx1: 变换后特征图 [B, C, H, W]
        selected_channels: 指定要保存的通道列表
        save_dir: 保存目录
    """

    os.makedirs(save_dir, exist_ok=True)
    colormap = cm.get_cmap('viridis')  # 可改为 'plasma', 'inferno', 'magma' 等

    for channel_idx in selected_channels:
        # 保存原始特征图通道
        if fx1 is not None:
            channel_data = fx1[0, channel_idx].detach().cpu().numpy()
            min_val = channel_data.min()
            max_val = channel_data.max()
            if max_val > min_val:
                norm_data = (channel_data - min_val) / (max_val - min_val)
            else:
                norm_data = np.zeros_like(channel_data)
            color_mapped = colormap(norm_data)[:, :, :3]  # 去掉 alpha 通道
            color_mapped = (color_mapped * 255).astype(np.uint8)
            channel_image = Image.fromarray(color_mapped)
            channel_image.save(os.path.join(save_dir, f'original_channel_{channel_idx}.png'))

        # 保存变换后特征图通道
        if warped_fx1 is not None:
            channel_data = warped_fx1[0, channel_idx].detach().cpu().numpy()
            min_val = channel_data.min()
            max_val = channel_data.max()
            if max_val > min_val:
                norm_data = (channel_data - min_val) / (max_val - min_val)
            else:
                norm_data = np.zeros_like(channel_data)
            color_mapped = colormap(norm_data)[:, :, :3]
            color_mapped = (color_mapped * 255).astype(np.uint8)
            channel_image = Image.fromarray(color_mapped)
            channel_image.save(os.path.join(save_dir, f'warped_channel_{channel_idx}.png'))


if __name__ == '__main__':

    with open("./configs/test_fmb_flir.json", 'r') as fp:
        cfg = json.load(fp)

    ir_affine_deform_path = "../../dataset/FLIR/ctest/pad/ir/FLIR_09302.png"
    ir_pseudo_path = "../../dataset/FLIR/ctest/pad/vis/FLIR_09302.png"
    vis_path = "../../dataset/FLIR/ctest/pad/vis/FLIR_09302.png"
    save_dir = "./feature_visable/FLIR_09302/predict"
    save_ori_dir = "./feature_visable/FLIR_09302/ori"
    save_reg_dir = "./feature_visable/FLIR_09302/reg"

    ir_affine_deform_path = "../../dataset/FLIR/ctest/pad/ir/FLIR_00320.png"
    ir_pseudo_path = "../../dataset/FLIR/ctest/pad/vis/FLIR_00320.png"
    vis_path = "../../dataset/FLIR/ctest/pad/vis/FLIR_00320.png"
    save_dir = "./feature_visable/FLIR_00320/predict"
    save_ori_dir = "./feature_visable/FLIR_00320/ori"
    save_reg_dir = "./feature_visable/FLIR_00320/reg"

    ir_affine_deform_path = "../../dataset/FLIR/ctest/pad/ir/FLIR_05229.png"
    ir_pseudo_path = "../../dataset/FLIR/ctest/pad/vis/FLIR_05229.png"
    vis_path = "../../dataset/FLIR/ctest/pad/vis/FLIR_05229.png"
    save_dir = "./feature_visable/FLIR_05229/predict"
    save_ori_dir = "./feature_visable/FLIR_05229/ori"
    save_reg_dir = "./feature_visable/FLIR_05229/reg"

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_ori_dir, exist_ok=True)
    os.makedirs(save_reg_dir, exist_ok=True)

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

    im_to_tensor = transforms.Compose([
        transforms.ToTensor()
    ])

    ir_affine_deform = im_to_tensor(Image.open(ir_affine_deform_path).convert('RGB')).unsqueeze(0).to(device)
    ir_pseudo = im_to_tensor(Image.open(ir_pseudo_path).convert('RGB')).unsqueeze(0).to(device)
    vis = im_to_tensor(Image.open(vis_path)).unsqueeze(0).to(device)

    model = RegSeg(cfg)
    model = load_ckpt(cfg['load_ckpt'], model, cfg['gpu'])
    model.to(device)

    warped_x, rewarped_y, flowall, out, fx1_ori, warped_fx1 = model(ir_affine_deform, ir_pseudo, vis)

    # Save predict images
    predict = out.max(1)[1].cpu().numpy()  # [b, h, w]
    predict_RGB = class_to_RGB(predict[0], cfg['n_classes'], cmap)
    cv2.imwrite(os.path.join(save_dir, 'predict' + '.png'), predict[0])
    cv2.imwrite(os.path.join(save_dir, 'predict_rgb' + '.png'), predict_RGB[:, :, ::-1])

    selected_channels = [15]
    save_selected_channels_png(fx1_ori, None, selected_channels, save_ori_dir)

    warped_x, rewarped_y, flowall, out, fx1_seg, warped_fx1 = model(vis, ir_pseudo, vis)
    save_selected_channels_png(fx1_seg, None, selected_channels, save_reg_dir)
