import torch
from torch import nn
from einops import rearrange
import torch.nn.functional as F

from utils import UNPAD

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
        #TODO: self.channels = [96, 192, 384, 768]，self.channels // patch_size**2 = [6, 12, 24, 48]，可否根据每层channels动态定义nn.Conv2d的数量（==2，3，4）和in_channel、out_channel参数

    def forward(self, x, y):
        concat = torch.cat([x, y], dim=1)
        cost_vol = self.Conv1(concat)
        cost_vol = self.Conv2(cost_vol)
        cost_vol = self.Conv3(cost_vol)
        cost_vol = self.Conv4(cost_vol)
        flow = self.Conv5(cost_vol)

        return flow


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, dim, patch_size=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.expand = nn.Linear(dim, patch_size * patch_size * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H, W, C
        """
        x = self.expand(x)  # B, H, W, 16C
        B, H, W, C = x.shape
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.patch_size, p2=self.patch_size,
                      c=C // (self.patch_size ** 2))
        x = self.norm(x)
        return x


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


class Reg(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        if cfg['backbone'] == 'vmamba_tiny':
            try:
                from .VMamba.MambaEncoder import vssm_tiny as backbone, fusion_tiny as fusion
            except:
                from nets.VMamba.MambaEncoder import vssm_tiny as backbone, fusion_tiny as fusion
            self.reg_seg_backbone = backbone()
            self.channels = [96, 192, 384, 768]
        if cfg['backbone'] == 'vmamba_small':
            try:
                from .VMamba.MambaEncoder import vssm_small as backbone, fusion_small as fusion
            except:
                from nets.VMamba.MambaEncoder import vssm_small as backbone, fusion_small as fusion
            self.reg_seg_vbackbone = backbone()
            self.channels = [96, 192, 384, 768]
        if cfg['backbone'] == 'vmamba_base':
            try:
                from .VMamba.MambaEncoder import vssm_base as backbone, fusion_base as fusion
            except:
                from nets.VMamba.MambaEncoder import vssm_base as backbone, fusion_base as fusion
            self.reg_seg_backbone = backbone()
            self.channels = [128, 256, 512, 1024]
        self.patch_size = 4
        self.aux_head = None
        self.decode_head = MambaDecoder(img_size=[cfg["height"], cfg["width"]], in_channels=self.channels,
                                            num_classes=cfg["n_classes"], embed_dim=self.channels[0])
        # self.reg_block4 = RegBlock(self.channels[3], self.channels[3], self.channels[3] * 2)
        # self.reg_block3 = RegBlock(self.channels[2], self.channels[2], self.channels[2] * 2)
        # self.reg_block2 = RegBlock(self.channels[1], self.channels[1], self.channels[1] * 2)
        # self.reg_block1 = RegBlock(self.channels[0], self.channels[0], self.channels[0] * 2)
        self.reg_block = nn.ModuleList()
        for i in range(4):
            self.reg_block.append(RegBlock(self.channels[i] // self.patch_size**2, self.channels[i] // self.patch_size**2, self.channels[i] // self.patch_size**2 * 2))
        self.reg_transformer = nn.ModuleList()
        for i in range(4):
            self.reg_transformer.append(SpatialTransformer([s // 2**i for s in [cfg["height"], cfg["width"]]]))
        self.up_x2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.up_final = FinalUpsample_X4(dim=self.channels[0])
        # self.up_final = FinalPatchExpand_X4(dim=self.channels[0])

    def forward(self, x, y, vis):
        """
           Forward function.

           Parameters:
               x: moving_images, [batch_size, channel=3, H=576, W=768]
               y: fixed_images, [batch_size, channel=3, H=576, W=768]
               vis: visible_images, [batch_size, channel=3, H=576, W=768]
        """
        fx1, fx2, fx3, fx4 = self.reg_seg_backbone(x)
        fy1, fy2, fy3, fy4 = self.reg_seg_backbone(y)


        # range a
        B, C, H, W = fx4.shape
        fx4 = fx4.view(B, -1, self.patch_size * H, self.patch_size * W)
        fy4 = fy4.view(B, -1, self.patch_size * H, self.patch_size * W)
        wx4 = fx4
        flow = self.reg_block[3](wx4, fy4)
        flowall = flow
        # warped_fx4 = self.transformer[0](fx4, flowall)
        # rewarped_fy4 = self.transformer[0](fy4, -flowall)
        # fx4 = fx4.view(B, -1, H // self.patch_size, W // self.patch_size)
        # fy4 = fy4.view(B, -1, H // self.patch_size, W // self.patch_size)
        # warped_fx4 = warped_fx4.view(B, -1, H // self.patch_size, W // self.patch_size)

        # range b
        B, C, H, W = fx3.shape
        fx3 = fx3.view(B, -1, self.patch_size * H, self.patch_size * W)
        fy3 = fy3.view(B, -1, self.patch_size * H, self.patch_size * W)
        flowall = self.up_x2(2 * flowall)
        wx3 = self.reg_transformer[2](fx3, flowall)
        flow = self.reg_block[2](wx3, fy3)
        flowall = self.reg_transformer[2](flowall, flow)
        # warped_fx3 = self.transformer[0](fx3, flowall)
        # rewarped_fy3 = self.transformer[0](fy3, -flowall)
        # fx3 = fx3.view(B, -1, H // self.patch_size, W // self.patch_size)
        # fy3 = fy3.view(B, -1, H // self.patch_size, W // self.patch_size)
        # warped_fx3 = warped_fx3.view(B, -1, H // self.patch_size, W // self.patch_size)

        # range c
        B, C, H, W = fx2.shape
        fx2 = fx2.view(B, -1, self.patch_size * H, self.patch_size * W)
        fy2 = fy2.view(B, -1, self.patch_size * H, self.patch_size * W)
        flowall = self.up_x2(2 * flowall)
        wx2 = self.reg_transformer[1](fx2, flowall)
        flow = self.reg_block[1](wx2, fy2)
        flowall = self.reg_transformer[1](flowall, flow)
        # warped_fx2 = self.transformer[0](fx2, flowall)
        # rewarped_fy2 = self.transformer[0](fy2, -flowall)
        # fx2 = fx2.view(B, -1, H // self.patch_size, W // self.patch_size)
        # fy2 = fy2.view(B, -1, H // self.patch_size, W // self.patch_size)
        # warped_fx2 = warped_fx2.view(B, -1, H // self.patch_size, W // self.patch_size)

        # range d
        B, C, H, W = fx1.shape
        fx1 = fx1.view(B, -1, self.patch_size * H, self.patch_size * W)
        fy1 = fy1.view(B, -1, self.patch_size * H, self.patch_size * W)
        flowall = self.up_x2(2 * flowall)
        wx1 = self.reg_transformer[0](fx1, flowall)
        flow = self.reg_block[0](wx1, fy1)
        flowall = self.reg_transformer[0](flowall, flow)
        # warped_fx1 = self.transformer[0](fx1, flowall)
        # rewarped_fy1 = self.transformer[0](fy1, -flowall)
        # fx1 = fx1.view(B, -1, H // self.patch_size, W // self.patch_size)
        # fy1 = fy1.view(B, -1, H // self.patch_size, W // self.patch_size)
        # warped_fx1 = warped_fx1.view(B, -1, H // self.patch_size, W // self.patch_size)

        warped_x = self.reg_transformer[0](x, flowall)
        rewarped_y = self.reg_transformer[0](y, -flowall)

        return warped_x, rewarped_y, flowall


class RegSeg(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        if cfg['backbone'] == 'vmamba_tiny':
            try:
                from .VMamba.MambaEncoder import vssm_tiny as backbone, fusion_tiny as fusion
            except:
                from nets.VMamba.MambaEncoder import vssm_tiny as backbone, fusion_tiny as fusion
            self.reg_seg_backbone = backbone()
            self.seg_backbone = backbone()
            self.seg_fusion = fusion()
            self.channels = [96, 192, 384, 768]
        if cfg['backbone'] == 'vmamba_small':
            try:
                from .VMamba.MambaEncoder import vssm_small as backbone, fusion_small as fusion
            except:
                from nets.VMamba.MambaEncoder import vssm_small as backbone, fusion_small as fusion
            self.reg_seg_backbone = backbone()
            self.seg_backbone = backbone()
            self.seg_fusion = fusion()
            self.channels = [96, 192, 384, 768]
        if cfg['backbone'] == 'vmamba_base':
            try:
                from .VMamba.MambaEncoder import vssm_base as backbone, fusion_base as fusion
            except:
                from nets.VMamba.MambaEncoder import vssm_base as backbone, fusion_base as fusion
            self.reg_seg_backbone = backbone()
            self.seg_backbone = backbone()
            self.seg_fusion = fusion()
            self.channels = [128, 256, 512, 1024]
        self.pad_height = cfg["pad_height"]
        self.pad_width = cfg["pad_width"]
        self.height = cfg["height"]
        self.width = cfg["width"]
        self.orisize = [self.height-self.pad_height*2, self.width-self.pad_width*2]
        self.patch_size = 4
        self.seg_decode_head = MambaDecoder(img_size=[self.height-self.pad_height*2, self.width-self.pad_width*2],
                                        in_channels=self.channels, num_classes=cfg["n_classes"], embed_dim=self.channels[0])
        self.reg_block = nn.ModuleList()
        for i in range(4):
            self.reg_block.append(
                RegBlock(self.channels[i] // self.patch_size ** 2, self.channels[i] // self.patch_size ** 2,
                         self.channels[i] // self.patch_size ** 2 * 2))
        self.reg_transformer = nn.ModuleList()
        for i in range(4):
            self.reg_transformer.append(SpatialTransformer([s // 2 ** i for s in [self.height, self.width]]))
        self.up_x2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.up_final = FinalUpsample_X4(dim=self.channels[0])
        # self.up_final = FinalPatchExpand_X4(dim=self.channels[0])
        self.unpad = nn.ModuleList()
        for i in range(4):
            self.unpad.append(UNPAD(self.pad_height // 2**i, self.pad_width // 2**i))

    def forward(self, x, y, vis):
        """
           Forward function.

           Parameters:
               x: moving_images, [batch_size, channel=3, H=576, W=768]
               y: fixed_images, [batch_size, channel=3, H=576, W=768]
               vis: visible_images, [batch_size, channel=3, H=576, W=768]
        """
        fx1, fx2, fx3, fx4 = self.reg_seg_backbone(x)
        fy1, fy2, fy3, fy4 = self.reg_seg_backbone(y)

        # # === NEW: cache native first-scale features BEFORE any view ===
        # fx1_native, fx2_native, fx3_native, fx4_native = fx1, fx2, fx3, fx4  # [B, C1, Hf, Wf]

        # Registration
        # range a
        B, C, H, W = fx4.shape
        fx4 = fx4.view(B, -1, self.patch_size * H, self.patch_size * W)
        fy4 = fy4.view(B, -1, self.patch_size * H, self.patch_size * W)
        wx4 = fx4
        flow = self.reg_block[3](wx4, fy4)
        flowall = flow
        warped_fx4 = self.reg_transformer[3](fx4, flowall)
        warped_fx4 = self.unpad[3](warped_fx4)
        warped_fx4 = warped_fx4.view(B, -1, H-self.pad_height*2//self.patch_size//2**3, W-self.pad_width*2//self.patch_size//2**3)

        # range b
        B, C, H, W = fx3.shape
        fx3 = fx3.view(B, -1, self.patch_size * H, self.patch_size * W)
        fy3 = fy3.view(B, -1, self.patch_size * H, self.patch_size * W)
        flowall = self.up_x2(2 * flowall)
        wx3 = self.reg_transformer[2](fx3, flowall)
        flow = self.reg_block[2](wx3, fy3)
        flowall = self.reg_transformer[2](flowall, flow)
        warped_fx3 = self.reg_transformer[2](fx3, flowall)
        warped_fx3 = self.unpad[2](warped_fx3)
        warped_fx3 = warped_fx3.view(B, -1, H-self.pad_height*2//self.patch_size//2**2, W-self.pad_width*2//self.patch_size//2**2)

        # range c
        B, C, H, W = fx2.shape
        fx2 = fx2.view(B, -1, self.patch_size * H, self.patch_size * W)
        fy2 = fy2.view(B, -1, self.patch_size * H, self.patch_size * W)
        flowall = self.up_x2(2 * flowall)
        wx2 = self.reg_transformer[1](fx2, flowall)
        flow = self.reg_block[1](wx2, fy2)
        flowall = self.reg_transformer[1](flowall, flow)
        warped_fx2 = self.reg_transformer[1](fx2, flowall)
        warped_fx2 = self.unpad[1](warped_fx2)
        warped_fx2 = warped_fx2.view(B, -1, H-self.pad_height*2//self.patch_size//2, W-self.pad_width*2//self.patch_size//2)

        # range d
        B, C, H, W = fx1.shape
        fx1 = fx1.view(B, -1, self.patch_size * H, self.patch_size * W)
        fy1 = fy1.view(B, -1, self.patch_size * H, self.patch_size * W)
        flowall = self.up_x2(2 * flowall)
        wx1 = self.reg_transformer[0](fx1, flowall)
        flow = self.reg_block[0](wx1, fy1)
        flowall = self.reg_transformer[0](flowall, flow)
        warped_fx1 = self.reg_transformer[0](fx1, flowall)
        warped_fx1 = self.unpad[0](warped_fx1)
        warped_fx1 = warped_fx1.view(B, -1, H-self.pad_height*2//self.patch_size, W-self.pad_width*2//self.patch_size)

        warped_x = self.reg_transformer[0](x, flowall)
        rewarped_y = self.reg_transformer[0](y, -flowall)

        ir = [warped_fx1, warped_fx2, warped_fx3, warped_fx4]
        vis = self.unpad[0](vis)
        vis = self.seg_backbone(vis)

        # Segmentation
        fet = self.seg_fusion(vis, ir)
        out = self.seg_decode_head.forward(fet)
        out = F.interpolate(out, size=self.orisize, mode='bilinear', align_corners=False)

        # # === NEW: warp first-scale native features with matched transformer and resized flow ===
        # # pick transformer whose grid matches fx1_native spatial size
        # Hf, Wf = fx1_native.shape[-2], fx1_native.shape[-1]
        # # find i such that [height // 2**i, width // 2**i] == (Hf, Wf)
        # # assuming cfg is available in class, otherwise compute i from current transformer sizes
        # tgt_idx = None
        # for i in range(4):
        #     H_i, W_i = self.reg_transformer[i].size  # assuming SpatialTransformer(size=[H_i,W_i]) stores 'size'
        #     if H_i == Hf and W_i == Wf:
        #         tgt_idx = i
        #         break
        # if tgt_idx is None:
        #     # fallback: choose the transformer whose size is closest to fx1_native
        #     sizes = [self.reg_transformer[i].size for i in range(4)]
        #     diffs = [abs(s[0] - Hf) + abs(s[1] - Wf) for s in sizes]
        #     tgt_idx = int(diffs.index(min(diffs)))
        # # resize flow to feature resolution and scale magnitude
        # Hi, Wi = x.shape[-2], x.shape[-1]
        # flow_feat = F.interpolate(flowall, size=(Hf, Wf), mode='bilinear', align_corners=True)
        # scale_x = Wf / float(Wi)
        # scale_y = Hf / float(Hi)
        # flow_feat[:, 0] = flow_feat[:, 0] * scale_x
        # flow_feat[:, 1] = flow_feat[:, 1] * scale_y
        # warped_fx1_out = self.reg_transformer[tgt_idx](fx1_native, flow_feat)
        # fx1_out = fx1_native
        # return warped_x, rewarped_y, flowall, out, fx1_out, warped_fx1_out

        return warped_x, rewarped_y, flowall, out


class Seg(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        if cfg['backbone'] == 'vmamba_tiny':
            try:
                from .VMamba.MambaEncoder import vssm_tiny as backbone, fusion_tiny as fusion
            except:
                from nets.VMamba.MambaEncoder import vssm_tiny as backbone, fusion_tiny as fusion
            self.reg_seg_backbone = backbone()
            self.seg_fusion = fusion()
            self.channels = [96, 192, 384, 768]
        if cfg['backbone'] == 'vmamba_small':
            try:
                from .VMamba.MambaEncoder import vssm_small as backbone, fusion_small as fusion
            except:
                from nets.VMamba.MambaEncoder import vssm_small as backbone, fusion_small as fusion
            self.reg_seg_backbone = backbone()
            self.seg_fusion = fusion()
            self.channels = [96, 192, 384, 768]
        if cfg['backbone'] == 'vmamba_base':
            try:
                from .VMamba.MambaEncoder import vssm_base as backbone, fusion_base as fusion
            except:
                from nets.VMamba.MambaEncoder import vssm_base as backbone, fusion_base as fusion
            self.reg_seg_backbone = backbone()
            self.seg_fusion = fusion()
            self.channels = [128, 256, 512, 1024]
        self.pad_height = cfg["pad_height"]
        self.pad_width = cfg["pad_width"]
        self.height = cfg["height"]
        self.width = cfg["width"]
        self.orisize = [self.height-self.pad_height*2, self.width-self.pad_width*2]
        self.patch_size = 4
        self.seg_decode_head = MambaDecoder(img_size=[self.height-self.pad_height*2, self.width-self.pad_width*2],
                                        in_channels=self.channels, num_classes=cfg["n_classes"], embed_dim=self.channels[0])
        self.reg_block = nn.ModuleList()
        for i in range(4):
            self.reg_block.append(
                RegBlock(self.channels[i] // self.patch_size ** 2, self.channels[i] // self.patch_size ** 2,
                         self.channels[i] // self.patch_size ** 2 * 2))
        self.reg_transformer = nn.ModuleList()
        for i in range(4):
            self.reg_transformer.append(SpatialTransformer([s // 2 ** i for s in [self.height, self.width]]))
        self.up_x2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.unpad = nn.ModuleList()
        for i in range(4):
            self.unpad.append(UNPAD(self.pad_height // 2**i, self.pad_width // 2**i))

    def forward(self, ir, vis):
        """
           Forward function.

           Parameters:
               ir: infrared_images, [batch_size, channel=3, H=576, W=768]
               vis: visible_images, [batch_size, channel=3, H=576, W=768]
        """
        ir = self.unpad[0](ir)
        ir = self.reg_seg_backbone(ir)
        vis = self.unpad[0](vis)
        vis = self.reg_seg_backbone(vis)

        # Segmentation
        fet = self.seg_fusion(vis, ir)
        out = self.seg_decode_head.forward(fet)
        out = F.interpolate(out, size=self.orisize, mode='bilinear', align_corners=False)

        return out



if __name__ == '__main__':
    import json
    cfg = None
    with open('../configs/train_mfnet.json', 'r') as fp:
        cfg = json.load(fp)
    assert cfg is not None
    num_minibatch = 1
    ir_affine_deform = torch.randn(num_minibatch, 3, 576, 768).cuda()
    ir_pseudo = torch.randn(num_minibatch, 3, 576, 768).cuda()
    vis = torch.randn(num_minibatch, 3, 576, 768).cuda()
    model1 = RegSeg(cfg).cuda(0)
    # for n, p in model.named_parameters():
    #     print(n)
    # warped_moving_ir, rewarped_fixed_ir_pseudo, flow, set_out = model(ir_affine_deform, ir_pseudo, vis)
    # print('end')

    from copy import deepcopy
    model2 = deepcopy(model1)
    from thop import profile, clever_format
    macs, params = profile(model1, inputs=(ir_affine_deform, ir_pseudo, vis))
    macs, params = clever_format([macs, params], "%.3f")
    print('thop ------- MACs: ', macs, 'Params: ', params)
    from calflops import calculate_flops
    flops, macs, params = calculate_flops(model2, args=[ir_affine_deform, ir_pseudo, vis], print_detailed=False, output_precision=4)
    print("calflops ------- FLOPs:%s   MACs:%s   Params:%s \n" % (flops, macs, params))
