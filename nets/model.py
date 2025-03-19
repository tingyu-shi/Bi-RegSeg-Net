import torch
from torch import nn
import torch.nn.functional as F

try:
    from .basic_module import SpatialTransformer
except:
    from nets.basic_module import SpatialTransformer

try:
    from .VMamba.MambaDecoder import MambaDecoder
except:
    from nets.VMamba.MambaDecoder import MambaDecoder

class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, in_channels, out_channels, kernal_size=3, stride=1, padding=1, alpha=0.1):
        super().__init__()

        self.main = nn.Conv2d(in_channels, out_channels, kernal_size, stride, padding)
        self.activation = nn.LeakyReLU(alpha)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out

class ConvResBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, in_channels, out_channels, kernal_size=3, stride=1, padding=1, alpha=0.1):
        super().__init__()

        self.main = nn.Conv2d(in_channels, out_channels, kernal_size, stride, padding)
        self.activation = nn.LeakyReLU(alpha)

    def forward(self, x):
        out = self.main(x)
        out = x + out
        out = self.activation(out)
        return out

class RegBlock(nn.Module):
    def __init__(self, x_channel, y_channel, out_channel):
        super(RegBlock, self).__init__()
        self.Conv1 = ConvBlock(x_channel+y_channel, out_channel)
        self.Conv2 = ConvResBlock(out_channel, out_channel)
        self.Conv3 = ConvResBlock(out_channel, out_channel)
        self.Conv4 = nn.Conv2d(out_channel, out_channel//2, 3, padding=1)
        self.Conv5 = nn.Conv2d(out_channel//2, 2, 3, padding=1)

    def forward(self, x, y):
        concat = torch.cat([x, y], dim=1)
        cost_vol = self.Conv1(concat)
        cost_vol = self.Conv2(cost_vol)
        cost_vol = self.Conv3(cost_vol)
        cost_vol = self.Conv4(cost_vol)
        flow = self.Conv5(cost_vol)

        return flow


class FinalUpsample_X4(nn.Module):
    def __init__(self, dim, patch_size=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.linear1 = nn.Linear(dim, dim, bias=False)
        self.linear2 = nn.Linear(dim, dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H, W, C
        """
        B, H, W, C = x.shape
        x = self.linear1(x).permute(0, 3, 1, 2).contiguous()  # B, C, H, W
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False).permute(0, 2, 3, 1).contiguous()  # B, 2H, 2W, C
        x = self.linear2(x).permute(0, 3, 1, 2).contiguous()  # B, C, 2H, 2W
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False).permute(0, 2, 3, 1).contiguous()  # B, 4H, 4W, C
        x = self.norm(x)
        return x

class RegSeg(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        if cfg['backbone'] == 'vmamba_tiny':
            try:
                from .VMamba.MambaEncoder import vssm_tiny as backbone, fusion_tiny as fusion
            except:
                from nets.VMamba.MambaEncoder import vssm_tiny as backbone, fusion_tiny as fusion
            self.backbone = backbone()
            self.fusion = fusion()
            self.channels = [96, 192, 384, 768]
        if cfg['backbone'] == 'vmamba_small':
            try:
                from .VMamba.MambaEncoder import vssm_small as backbone, fusion_small as fusion
            except:
                from nets.VMamba.MambaEncoder import vssm_small as backbone, fusion_small as fusion
            self.backbone = backbone()
            self.fusion = fusion()
            self.channels = [96, 192, 384, 768]
        if cfg['backbone'] == 'vmamba_base':
            try:
                from .VMamba.MambaEncoder import vssm_base as backbone, fusion_base as fusion
            except:
                from nets.VMamba.MambaEncoder import vssm_base as backbone, fusion_base as fusion
            self.backbone = backbone()
            self.fusion = fusion()
            self.channels = [128, 256, 512, 1024]
        self.aux_head = None
        self.decode_head = MambaDecoder(img_size=[cfg["height"], cfg["width"]], in_channels=self.channels,
                                            num_classes=cfg["n_classes"], embed_dim=self.channels[0])
        # self.reg_block4 = RegBlock(self.channels[3], self.channels[3], self.channels[3] * 2)
        # self.reg_block3 = RegBlock(self.channels[2], self.channels[2], self.channels[2] * 2)
        # self.reg_block2 = RegBlock(self.channels[1], self.channels[1], self.channels[1] * 2)
        # self.reg_block1 = RegBlock(self.channels[0], self.channels[0], self.channels[0] * 2)
        self.reg_block = nn.ModuleList()
        for i in range(4):
            self.reg_block.append(RegBlock(self.channels[i], self.channels[i], self.channels[i] * 2))
        self.transformer = nn.ModuleList()
        for i in range(2, 6):
            self.transformer.append(SpatialTransformer([s // 2**i for s in [cfg["height"], cfg["width"]]]))
        self.up_x2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_final = FinalUpsample_X4(dim=self.channels[0])

    def forward(self, x, y, vis):
        """
           Forward function.

           Parameters:
               x: moving_images, [batch_size, channel=3, H=576, W=768]
               y: fixed_images, [batch_size, channel=3, H=576, W=768]
               vi: visible_images, [batch_size, channel=3, H=576, W=768]
        """
        fx1, fx2, fx3, fx4 = self.backbone(x)
        fy1, fy2, fy3, fy4 = self.backbone(y)

        # range a
        wx4 = fx4
        flow = self.reg_block[3](wx4, fy4)
        flowall = flow

        # range b
        flowall = self.up_x2(2 * flowall)
        wx3 = self.transformer[2](fx3, flowall)
        flow = self.reg_block[2](wx3, fy3)
        flowall = self.transformer[2](flowall, flow) + flow

        # range c
        flowall = self.up_x2(2 * flowall)
        wx2 = self.transformer[1](fx2, flowall)
        flow = self.reg_block[1](wx2, fy2)
        flowall = self.transformer[1](flowall, flow) + flow

        # range d
        flowall = self.up_x2(2 * flowall)
        wx1 = self.transformer[0](fx1, flowall)
        flow = self.reg_block[0](wx1, fy1)
        flowall = self.transformer[0](flowall, flow) + flow

        # Final Upsample X4
        flowall = self.up_final(flowall)

        warped_x = self.transformer[0](x, flowall)
        rewarped_y = self.transformer[0](y, -flowall)

        return warped_x, rewarped_y, flowall

    # def encode_decode(self, rgb, modal_x):
    #     """Encode images with backbone and decode into a semantic segmentation
    #     map of the same size as input."""
    #     if not self.deep_supervision:
    #         orisize = rgb.shape
    #         x = self.backbone(rgb, modal_x)
    #         out = self.decode_head.forward(x)
    #         out = F.interpolate(out, size=orisize[2:], mode='bilinear', align_corners=False)
    #         if self.aux_head:
    #             aux_fm = self.aux_head(x[self.aux_index])
    #             aux_fm = F.interpolate(aux_fm, size=orisize[2:], mode='bilinear', align_corners=False)
    #             return out, aux_fm
    #         return out
    #
    # def forward(self, rgb, modal_x):
    #     if self.aux_head:
    #         out, aux_fm = self.encode_decode(rgb, modal_x)
    #     else:
    #         out = self.encode_decode(rgb, modal_x)
    #     return out

if __name__ == '__main__':
    import json
    cfg = None
    with open('../configs/train_sigmaseg_mfnet.json', 'r') as fp:
        cfg = json.load(fp)
    assert cfg is not None
    num_minibatch = 1
    ir_affine_deform = torch.randn(num_minibatch, 3, 576, 768).cuda()
    ir_pseudo = torch.randn(num_minibatch, 3, 576, 768).cuda()
    vi = torch.randn(num_minibatch, 3, 576, 768).cuda()
    model = RegSeg(cfg).cuda(0)
    warped_moving_ir, rewarped_fixed_ir_pseudo, flow = model(ir_affine_deform, ir_pseudo, vi)