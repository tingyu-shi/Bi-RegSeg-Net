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
    (0, 0, 0),  # 0 Unlabelled
    (64, 0, 128),  # 1 Car
    (64, 64, 0),  # 2 Person
    (0, 128, 192),  # 3 Bike
    (0, 0, 192),  # 4 Curve
    (128, 128, 0),  # 5 Car Stop
    (64, 64, 128),  # 6 Guardrail
    (192, 128, 128),  # 7 Color Cone
    (192, 64, 0)  # 8 Bump
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

    with open("./configs/test_mfnet.json", 'r') as fp:
        cfg = json.load(fp)

    # ir_affine_deform_path = "../../dataset/MFNet/ctest/pad/ir_affine_deform/01486D.png"
    # ir_pseudo_path = "../../dataset/MFNet/ctest/pad/ir_pseudo_stylebooth/01486D.png"
    # vis_path = "../../dataset/MFNet/ctest/pad/vis/01486D.png"
    # save_dir = "./feature_visable/01486D/severe/"
    # ir_affine_deform_path = "../../dataset/MFNet/ctest/pad/ir_affine_deform_slight/01486D.png"
    # ir_pseudo_path = "../../dataset/MFNet/ctest/pad/ir_pseudo_stylebooth/01486D.png"
    # vis_path = "../../dataset/MFNet/ctest/pad/vis/01486D.png"
    # save_dir = "./feature_visable/01486D/slight/"
    ir_affine_deform_path = "../../dataset/MFNet/ctest/pad/ir/01486D.png"
    ir_pseudo_path = "../../dataset/MFNet/ctest/pad/ir_pseudo_stylebooth/01486D.png"
    vis_path = "../../dataset/MFNet/ctest/pad/vis/01486D.png"
    save_dir = "./feature_visable/01486D/registered/"

    os.makedirs(save_dir, exist_ok=True)

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

    warped_x, rewarped_y, flowall, out, fx1, warped_fx1 = model(ir_affine_deform, ir_pseudo, vis)

    # Save predict images
    predict = out.max(1)[1].cpu().numpy()  # [b, h, w]
    predict_RGB = class_to_RGB(predict[0], cfg['n_classes'], cmap)
    cv2.imwrite(os.path.join(save_dir, 'predict' + '.png'), predict[0])
    cv2.imwrite(os.path.join(save_dir, 'predict_rgb' + '.png'), predict_RGB[:, :, ::-1])

    selected_channels = [27]
    save_selected_channels_png(fx1, warped_fx1, selected_channels, save_dir)
