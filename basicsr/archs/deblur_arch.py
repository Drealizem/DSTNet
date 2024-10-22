import torch
import time
from torch import nn as nn
from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import ResidualBlockNoBN, flow_warp, make_layer
from basicsr.archs.ChanDynamic_GMLP import CWGDN
import torch.nn.functional as F
from basicsr.archs.wave_tf import DWT, IWT
from basicsr.archs.wave_tf import HaarDownsampling
from basicsr.archs.kpn_pixel import IDynamicDWConv
from einops import rearrange


@ARCH_REGISTRY.register()
class Deblur(nn.Module):
    """
    视频去模糊模型，使用深度学习技术来恢复模糊视频中的清晰帧。

    Args:
        num_feat (int): 网络中特征图的通道数，默认为64。
        num_block (int): 网络中使用的残差块的数量，默认为15。
    """
    def __init__(self, num_feat=64, num_block=15):
        super().__init__()
        self.num_feat = num_feat

        # extractor & reconstruction
        self.feat_extractor = nn.Conv3d(3, num_feat, (1, 3, 3), 1, (0, 1, 1), bias=True)  # 特征提取器，用于从输入的模糊帧中提取特征
        self.recons = nn.Conv3d(num_feat, 3, (1, 3, 3), 1, (0, 1, 1), bias=True)  # 重建模块，用于将提取的特征图重建为清晰的帧

        # wave tf
        self.wave = HaarDownsampling(num_feat)  # 哈尔小波变换模块，用于在变换域中处理特征图
        self.x_wave_1_conv1 = nn.Conv2d(num_feat * 3, num_feat * 3, 1, 1, 0, groups=3)  # 用于处理小波变换后的特征图的卷积层
        self.x_wave_1_conv2 = nn.Conv2d(num_feat * 3, num_feat * 3, 1, 1, 0, groups=3)
        # wave pro
        self.x_wave_2_conv1 = nn.Conv2d(num_feat * 3, num_feat * 3, 1, 1, 0, groups=3)  # 另一组用于处理小波变换后的特征图的卷积层
        self.x_wave_2_conv2 = nn.Conv2d(num_feat * 3, num_feat * 3, 1, 1, 0, groups=3)

        # transformer # 变换域特征融合模块，使用 CWGDN 进行特征融合
        transformer_scale4 = []
        for _ in range(5):
            transformer_scale4.append(
                nn.Sequential(CWGDN(dim=num_feat, ffn_expansion_factor=4, bias=False, LayerNorm_type='WithBias'))
            )
        self.transformer_scale4 = nn.Sequential(*transformer_scale4)

        # propogation branch # 特征传播模块，用于在时间维度上传播特征
        self.forward_propagation = manual_conv3d_propagation_forward(num_feat, num_block)
        self.backward_propagation = manual_conv3d_propagation_backward(num_feat, num_block)

        # activation functions # 激活函数
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, lrs):
        """
        模型的前向传播函数，处理输入的模糊视频帧，并输出去模糊后的帧。
        Args:
            lrs (Tensor): 输入的模糊视频帧，形状为 (batch_size, time, channel, height, width)。
        Returns:
            Tensor: 去模糊后的视频帧，形状为 (batch_size, time, channel, height, width)。
        """
        # time_start = time.time()
        # print(lrs.size())
        b, t, c, h, w = lrs.size()  # 时间维度上的特征提取
        lrs_feature = self.feat_extractor(rearrange(lrs, 'b t c h w -> b c t h w'))     # b c t h w
        # scale1 # 第一层小波变换
        tf_input_feature = rearrange(lrs_feature, 'b c t h w -> (b t) c h w')
        tf_wave1_l, tf_wave1_h = self.wave(tf_input_feature)
        tf_wave1_h = self.x_wave_1_conv2(self.lrelu(self.x_wave_1_conv1(tf_wave1_h)))
        # scale2 # 第二层小波变换
        tf_wave2_l, tf_wave2_h = self.wave(tf_wave1_l)
        tf_wave2_l = rearrange(self.transformer_scale4(rearrange(tf_wave2_l, '(b t) c h w -> b t c h w', b=b)), 'b t c h w -> (b t) c h w')
        tf_wave2_h = self.x_wave_2_conv2(self.lrelu(self.x_wave_2_conv1(tf_wave2_h)))
        # scale1 # 特征融合
        tf_wave1_l = self.wave(torch.cat([tf_wave2_l, tf_wave2_h], dim=1), rev=True)
        tf_wave1_l = rearrange(self.forward_propagation(self.backward_propagation(rearrange(tf_wave1_l, '(b t) c h w -> b t c h w', b=b))), 'b t c h w -> (b t) c h w')
        pro_feat = rearrange(self.wave(torch.cat([tf_wave1_l, tf_wave1_h], dim=1), rev=True), '(b t) c h w -> b t c h w', b=b)  # 重建去模糊帧

        # reconstruction # 输出去模糊后的帧
        out = rearrange(self.recons(rearrange(pro_feat, 'b t c h w -> b c t h w')), 'b c t h w -> b t c h w')

        # time_end = time.time()
        # print("inference time:", time_end - time_start)
        return out.contiguous() + lrs

'''
包含多个残差块的类，用于提取特征
    num_feat: 特征图的通道数。
    num_block: 残差块的数量。
    make_layer: 创建指定数量的残差块。
    forward: 将输入特征传递给残差块序列。
'''
class ResidualBlocks2D(nn.Module):
    def __init__(self, num_feat=64, num_block=30):
        super().__init__()
        self.main = nn.Sequential(
            make_layer(ResidualBlockNoBN, num_block, num_feat=num_feat))

    def forward(self, fea):
        return self.main(fea)


'''
实现了一个3D卷积层，用于在时间维度上向后传播特征
    num_feat: 特征图的通道数。
    num_block: 残差块的数量。
    conv1, conv2, conv3: 用于特征融合和处理的卷积层。
    kernel_conv_pixel: 动态像素卷积层。
    lrelu: 激活函数。
    resblock_bcakward2d: 2D残差块。
    forward: 实现特征的向后传播。
'''
class manual_conv3d_propagation_backward(nn.Module):
    def __init__(self, num_feat=64, num_block=15):
        super().__init__()
        self.num_feat = num_feat
        self.conv1 = nn.Conv2d(num_feat * 2, num_feat * 2, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.kernel_conv_pixel = IDynamicDWConv(num_feat, 3, 1, 4, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.resblock_bcakward2d = ResidualBlocks2D(num_feat, num_block)

    def forward(self, feature):
        # predefine
        b, t, c, h, w = feature.size()                           # b t 64 256 256
        backward_list = []
        feat_prop = feature.new_zeros(b, c, h, w)
        # propagation
        for i in range(t - 1, -1, -1):
            x_feat = feature[:, i, :, :, :]
            # fusion propagation
            feat_fusion = torch.cat([x_feat, feat_prop], dim=1)       # b 128 256 256
            feat_fusion = self.lrelu(self.conv1(feat_fusion))   # b 128 256 256
            feat_prop1, feat_prop2 = torch.split(feat_fusion, self.num_feat, dim=1)
            feat_prop1 = feat_prop1 * torch.sigmoid(self.conv2(feat_prop1))
            feat_prop2 = feat_prop2 * torch.sigmoid(self.conv3(feat_prop2))
            feat_prop = feat_prop1 + feat_prop2
            # dynamic conv
            feat_prop = self.kernel_conv_pixel(feat_prop)
            # resblock2D
            feat_prop = self.resblock_bcakward2d(feat_prop)
            backward_list.append(feat_prop)

        backward_list = backward_list[::-1]
        conv3d_feature = torch.stack(backward_list, dim=1)      # b 64 t 256 256
        return conv3d_feature

'''
实现了一个3D卷积层，用于在时间维度上向前传播特征。
    num_feat: 特征图的通道数。
    num_block: 残差块的数量。
    conv1, conv2, conv3: 用于特征融合和处理的卷积层。
    kernel_conv_pixel: 动态像素卷积层。
    lrelu: 激活函数。
    resblock_bcakward2d: 2D残差块。
    forward: 实现特征的向前传播。
'''
class manual_conv3d_propagation_forward(nn.Module):
    def __init__(self, num_feat=64, num_block=15):
        super().__init__()
        self.num_feat = num_feat
        self.conv1 = nn.Conv2d(num_feat * 2, num_feat * 2, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.kernel_conv_pixel = IDynamicDWConv(num_feat, 3, 1, 4, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.resblock_bcakward2d = ResidualBlocks2D(num_feat, num_block)

    def forward(self, feature):
        # predefine
        b, t, c, h, w = feature.size()                          # b t 64 256 256
        forward_list = []
        feat_prop = feature.new_zeros(b, c, h, w)
        for i in range(0, t):
            x_feat = feature[:, i, :, :, :]
            # fusion propagation
            feat_fusion = torch.cat([x_feat, feat_prop], dim=1)  # b 128 256 256
            feat_fusion = self.lrelu(self.conv1(feat_fusion))  # b 128 256 256
            feat_prop1, feat_prop2 = torch.split(feat_fusion, self.num_feat, dim=1)
            feat_prop1 = feat_prop1 * torch.sigmoid(self.conv2(feat_prop1))
            feat_prop2 = feat_prop2 * torch.sigmoid(self.conv3(feat_prop2))
            feat_prop = feat_prop1 + feat_prop2
            # dynamic conv
            feat_prop = self.kernel_conv_pixel(feat_prop)
            # resblock2D
            feat_prop = self.resblock_bcakward2d(feat_prop)
            forward_list.append(feat_prop)

        conv3d_feature = torch.stack(forward_list, dim=1)      # b 64 t 256 256
        return conv3d_feature
