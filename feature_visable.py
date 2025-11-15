import json
import os
import random

import cv2
import torch
import torchvision
from PIL import Image
from torchvision.transforms import transforms
from matplotlib import pyplot as plt

from nets.model import RegSeg, Reg
from utils import load_ckpt

with open("./configs/test_fmb.json", 'r') as fp:
    cfg = json.load(fp)

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

ir_affine_deform_path = "../../dataset/FMB/ctest/pad/ir_affine_deform/00129.png"
ir_pseudo_path = "../../dataset/FMB/ctest/pad/ir_pseudo_stylebooth/00129.png"
vis_path = "../../dataset/FMB/ctest/pad/vis/00129.png"
ir_affine_deform_path = "../../dataset/FMB/ctest/pad/ir_affine_deform/00359.png"
ir_pseudo_path = "../../dataset/FMB/ctest/pad/ir_pseudo_stylebooth/00359.png"
vis_path = "../../dataset/FMB/ctest/pad/vis/00359.png"

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

fx1 = fx1.data.squeeze(0)
warped_fx1 = warped_fx1.data.squeeze(0)

def random_num(size, end):
    range_ls = [i for i in range(end)]
    num_ls = []
    for i in range(size):
        num = random.choice(range_ls)
        range_ls.remove(num)
        num_ls.append(num)
    return num_ls

# 随机选取25个通道的特征图
print('warped_fx1 number:', warped_fx1.shape[0])
channel_num = random_num(25, warped_fx1.shape[0])
plt.figure(figsize=(10, 10))
for index, channel in enumerate(channel_num):
    ax = plt.subplot(5, 5, index + 1)
    plt.imshow(fx1[channel, :, :].cpu().numpy() )  # 灰度图参数cmap="gray"
plt.savefig("fx1_feature.jpg", dpi=300)

plt.figure(figsize=(10, 10))
for index, channel in enumerate(channel_num):
    ax = plt.subplot(5, 5, index + 1)
    plt.imshow(warped_fx1[channel, :, :].cpu().numpy() )  # 灰度图参数cmap="gray"
plt.savefig("warped_fx1_feature.jpg", dpi=300)