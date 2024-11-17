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

# 将模型注册到全局架构注册表中，方便后续通过名称获取模型实例
@ARCH_REGISTRY.register()
class Deblur(nn.Module):    # 本文使用的就是这个模型
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
        self.feat_extractor = nn.Conv3d(3, num_feat, (1, 3, 3), 1, (0, 1, 1), bias=True)    # 特征提取器，用于从输入的模糊帧中提取特征
        self.recons = nn.Conv3d(num_feat, 3, (1, 3, 3), 1, (0, 1, 1), bias=True)    # 重建模块，用于将提取的特征图重建为清晰的帧

        # wave tf # 哈尔小波变换模块，用于在变换域中处理特征图
        self.wave = HaarDownsampling(num_feat)
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
            lrs (Tensor): 输入的模糊视频帧（原始图像），形状为 (batch_size, time, channel, height, width)。

        Returns:
            Tensor: 去模糊后的视频帧，形状为 (batch_size, time, channel, height, width)。
        """
        # time_start = time.time()
        # print(lrs.size())
        b, t, c, h, w = lrs.size()  # 获取输入数据的维度信息，包括批次大小 b，时间维度 t，通道数 c，高度 h，宽度 w
        lrs_feature = self.feat_extractor(rearrange(lrs, 'b t c h w -> b c t h w'))     # b c t h w
        # scale1 # 使用特征提取器从输入的低分辨率图像中提取特征
        tf_input_feature = rearrange(lrs_feature, 'b c t h w -> (b t) c h w')
        tf_wave1_l, tf_wave1_h = self.wave(tf_input_feature)    # 第一次小波变换(分别输出低频和高频部分)
        tf_wave1_h = self.x_wave_1_conv2(self.lrelu(self.x_wave_1_conv1(tf_wave1_h)))   # 使用卷积层处理小波变换后的高频部分
        # scale2 # 对低频部分再次进行小波变换
        tf_wave2_l, tf_wave2_h = self.wave(tf_wave1_l)  # 第二次小波变换
        tf_wave2_l = rearrange(self.transformer_scale4(rearrange(tf_wave2_l, '(b t) c h w -> b t c h w', b=b)), 'b t c h w -> (b t) c h w') # 使用变换域特征融合模块(包含CWGDN)处理第二次小波变换后的低频部分
        tf_wave2_h = self.x_wave_2_conv2(self.lrelu(self.x_wave_2_conv1(tf_wave2_h)))   # 处理小波变换后的高频部分
        # scale1 # 将两次小波变换的结果进行融合
        tf_wave1_l = self.wave(torch.cat([tf_wave2_l, tf_wave2_h], dim=1), rev=True) # rev=True表示逆变换IWT；这里的tf_wave2_l已经完成了CWGDN；IWT的结果存入tf_wave1_l中
        # ***重点*** 这里就是WaveletFP模块（这里时间上先向后再向前传播）
        tf_wave1_l = rearrange(self.forward_propagation(self.backward_propagation(rearrange(tf_wave1_l, '(b t) c h w -> b t c h w', b=b))), 'b t c h w -> (b t) c h w') # 使用特征传播模块在时间维度上传播特征
        '''可将上一行替换成别的模块'''

        # 将传播后的特征与高频部分结合，进行小波逆变换，重建去模糊帧
        pro_feat = rearrange(self.wave(torch.cat([tf_wave1_l, tf_wave1_h], dim=1), rev=True), '(b t) c h w -> b t c h w', b=b)

        # reconstruction # 使用重建模块将特征图重建为去模糊后的图像
        out = rearrange(self.recons(rearrange(pro_feat, 'b t c h w -> b c t h w')), 'b c t h w -> b t c h w')

        # time_end = time.time()
        # print("inference time:", time_end - time_start)
        return out.contiguous() + lrs   # 返回去模糊后的图像，加上原始输入图像（可能是为了计算损失时使用）


class ResidualBlocks2D(nn.Module):
    """
    包含多个残差块的类，用于提取特征。
    Args:
        num_feat (int): 特征图的通道数。
        num_block (int): 残差块的数量。
    """
    def __init__(self, num_feat=64, num_block=30):
        """
        初始化 ResidualBlocks2D 类的实例。
        参数:
        num_feat (int): 每个残差块的输入和输出特征图的通道数。
        num_block (int): 残差块的数量。
        """
        super().__init__()
        self.main = nn.Sequential(
            make_layer(ResidualBlockNoBN, num_block, num_feat=num_feat))

    def forward(self, fea):
        """
        前向传播函数，将输入特征传递给残差块序列。
        参数:
        fea (Tensor): 输入特征图。
        返回:
        Tensor: 经过残差块处理后的特征图。
        """
        return self.main(fea)


class manual_conv3d_propagation_backward(nn.Module):
    """
    实现了一个3D卷积层，用于在时间维度上向后传播特征。

    这个模块特别设计用于处理视频数据，通过在时间维度上反向传播特征，
    来增强模型对时间序列中特征的理解，从而提高视频去模糊的效果。

    Args:
        num_feat (int): 特征图的通道数，决定了输入和输出特征图的通道数。
        num_block (int): 残差块的数量，用于构建残差块序列。
    """
    def __init__(self, num_feat=64, num_block=15):
        super().__init__()
        self.num_feat = num_feat
        # 定义三个卷积层，用于特征融合和处理
        self.conv1 = nn.Conv2d(num_feat * 2, num_feat * 2, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)

        self.kernel_conv_pixel = IDynamicDWConv(num_feat, 3, 1, 4, 1)    # 动态像素卷积层，用于处理特征图
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True) # 激活函数，使用LeakyReLU引入非线性
        self.resblock_bcakward2d = ResidualBlocks2D(num_feat, num_block)    # 2D残差块，用于提取特征
    '''
    manual_conv3d_propagation_backward 类继承自 nn.Module，是一个 PyTorch 模块，用于在时间维度上向后传播特征。
    num_feat 参数指定了特征图的通道数。
    num_block 参数指定了残差块的数量。
    self.conv1、self.conv2 和 self.conv3 是三个卷积层，用于特征融合和处理。
    self.kernel_conv_pixel 是一个动态像素卷积层，用于处理特征图。
    self.lrelu 是一个激活函数，使用 LeakyReLU 引入非线性。
    self.resblock_bcakward2d 是一个包含多个2D残差块的模块。
    '''

    def forward(self, feature):
        """
        前向传播函数，实现特征的向后传播。
        Args:
            feature (Tensor): 输入特征图，形状为 (batch_size, time, channel, height, width)。

        Returns:
            Tensor: 经过向后传播处理的特征图，形状为 (batch_size, time, channel, height, width)。
        """
        # predefine # 获取输入特征图的维度信息 # b: batch_size, t: time, c: channel, h: height, w: width
        b, t, c, h, w = feature.size()                           # b t 64 256 256
        backward_list = []  # 初始化一个空列表，用于存储每个时间步的特征
        feat_prop = feature.new_zeros(b, c, h, w)   # 初始化一个特征图，用于存储传播的特征
        # propagation   # 从后向前遍历每个时间步，实现特征的反向传播
        '''这里的for循环是整个模块的核心部分，通过从后向前遍历每个时间步，实现特征的反向传播。feat_prop 是一个初始化为零的特征图，用于存储传播的特征。'''
        for i in range(t - 1, -1, -1): # t-1到0，步长为-1
            x_feat = feature[:, i, :, :, :]   # b 64 256 256  # 获取当前时间步的特征，feature[:, i, :, :, :]表示取feature的第i个时间步的特征
            # fusion propagation
            feat_fusion = torch.cat([x_feat, feat_prop], dim=1)       # b 128 256 256   # 将当前时间步的特征与传播的特征进行融合
            feat_fusion = self.lrelu(self.conv1(feat_fusion))   # b 128 256 256     # 通过卷积层和激活函数处理融合后的特征
            feat_prop1, feat_prop2 = torch.split(feat_fusion, self.num_feat, dim=1) # 将融合后的特征分割成两部分，并分别通过卷积层和激活函数处理
            feat_prop1 = feat_prop1 * torch.sigmoid(self.conv2(feat_prop1))
            feat_prop2 = feat_prop2 * torch.sigmoid(self.conv3(feat_prop2))
            feat_prop = feat_prop1 + feat_prop2 # 将处理后的特征进行融合，得到传播的特征
            # dynamic conv
            feat_prop = self.kernel_conv_pixel(feat_prop)   # 通过动态像素卷积层处理传播的特征
            # resblock2D
            feat_prop = self.resblock_bcakward2d(feat_prop) # 通过2D残差块处理传播的特征
            backward_list.append(feat_prop) # 将处理后的特征添加到列表中

        backward_list = backward_list[::-1] # 将列表中的特征进行反转，以匹配时间维度的顺序
        conv3d_feature = torch.stack(backward_list, dim=1)      # b 64 t 256 256    # 将列表中的特征堆叠起来，形成最终的输出特征图
        return conv3d_feature
    '''
    feature 参数是输入特征图，形状为 (batch_size, time, channel, height, width)。
    b、t、c、h、w 分别表示批次大小、时间维度、通道数、高度和宽度。
    backward_list 是一个空列表，用于存储每个时间步的特征。
    feat_prop 是一个初始化为零的特征图，用于存储传播的特征。
    通过从后向前遍历每个时间步，实现特征的反向传播。
    feat_fusion 是将当前时间步的特征与传播的特征进行融合。
    feat_fusion 通过卷积层和激活函数处理。
    feat_prop1 和 feat_prop2 是将融合后的特征分割成两部分，并分别通过卷积层和激活函数处理。
    feat_prop 是将处理后的特征进行融合，得到传播的特征。
    feat_prop 通过动态像素卷积层处理。
    feat_prop 通过2D残差块处理。
    backward_list 将处理后的特征添加到列表中。
    backward_list 进行反转，以匹配时间维度的顺序。
    conv3d_feature 将列表中的特征堆叠起来，形成最终的输出特征图。
    '''


class manual_conv3d_propagation_forward(nn.Module):
    """
    实现了一个3D卷积层，用于在时间维度上向前传播特征。

    这个模块特别设计用于处理视频数据，通过在时间维度上正向传播特征，
    来增强模型对时间序列中特征的理解，从而提高视频去模糊的效果。

    Args:
        num_feat (int): 特征图的通道数，决定了输入和输出特征图的通道数。
        num_block (int): 残差块的数量，用于构建残差块序列。
    """
    def __init__(self, num_feat=64, num_block=15):
        super().__init__()
        self.num_feat = num_feat
        # 定义三个卷积层，用于特征融合和处理
        self.conv1 = nn.Conv2d(num_feat * 2, num_feat * 2, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.kernel_conv_pixel = IDynamicDWConv(num_feat, 3, 1, 4, 1)   # DTFF模块
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True) # 激活函数，使用LeakyReLU引入非线性
        self.resblock_bcakward2d = ResidualBlocks2D(num_feat, num_block)    # 2D残差块，用于提取特征

    '''
    manual_conv3d_propagation_forward 类继承自 nn.Module，是一个 PyTorch 模块，用于在时间维度上向前传播特征。
    num_feat 参数指定了特征图的通道数。
    num_block 参数指定了残差块的数量。
    类中定义了三个卷积层 conv1、conv2 和 conv3，用于特征融合和处理。
    IDynamicDWConv 是一个动态像素卷积层，用于处理特征图。
    LeakyReLU 激活函数用于引入非线性。
    ResidualBlocks2D 是一个包含多个2D残差块的模块。
    '''

    def forward(self, feature):
        """
        前向传播函数，实现特征的向前传播。
        Args:
            feature (Tensor): 输入特征图，形状为 (batch_size, time, channel, height, width)。

        Returns:
            Tensor: 经过向前传播处理的特征图，形状为 (batch_size, time, channel, height, width)。
        """
        # predefine # 获取输入特征图的维度信息
        b, t, c, h, w = feature.size()        # b t 64 256 256
        forward_list = []   # 初始化一个空列表，用于存储每个时间步的特征。用法：forward_list.append(feat_prop)
        feat_prop = feature.new_zeros(b, c, h, w)   # 初始化一个特征图，用于存储传播的特征
        '''通过从前向后遍历每个时间步，实现特征的正向传播。feat_prop是一个初始化为零的特征图，用于存储传播的特征。forward_list是一个空列表，用于存储所有时间步的特征。'''
        for i in range(0, t):   # 从前向后遍历每个时间步，实现特征的正向传播
            x_feat = feature[:, i, :, :, :] # 获取当前时间步的特征
            # fusion propagation
            feat_fusion = torch.cat([x_feat, feat_prop], dim=1)  # b 128 256 256    # 将当前时间步的特征与传播的特征进行融合
            feat_fusion = self.lrelu(self.conv1(feat_fusion))  # b 128 256 256  # 通过卷积层和激活函数处理融合后的特征
            feat_prop1, feat_prop2 = torch.split(feat_fusion, self.num_feat, dim=1) # 将融合后的特征分割成两部分，并分别通过卷积层和激活函数处理
            feat_prop1 = feat_prop1 * torch.sigmoid(self.conv2(feat_prop1)) # Element-wise Product
            feat_prop2 = feat_prop2 * torch.sigmoid(self.conv3(feat_prop2))
            feat_prop = feat_prop1 + feat_prop2 # 将处理后的特征进行融合，得到传播的特征
            # dynamic conv
            feat_prop = self.kernel_conv_pixel(feat_prop)   # 通过动态像素卷积层处理传播的特征
            # resblock2D
            feat_prop = self.resblock_bcakward2d(feat_prop) # 通过2D残差块处理传播的特征
            forward_list.append(feat_prop)  # 将处理后的特征添加到列表中

        # 将列表中的特征堆叠起来，形成最终的输出特征图
        conv3d_feature = torch.stack(forward_list, dim=1)      # b 64 t 256 256
        return conv3d_feature
    '''
    feature 参数是输入特征图，形状为 (batch_size, time, channel, height, width)。
    b、t、c、h、w 分别表示批次大小、时间维度、通道数、高度和宽度。
    forward_list 是一个空列表，用于存储每个时间步的特征。
    feat_prop 是一个初始化为零的特征图，用于存储传播的特征。
    通过从前向后遍历每个时间步，实现特征的正向传播。
    feat_fusion 是将当前时间步的特征与传播的特征进行融合。
    feat_fusion 通过卷积层和激活函数处理。
    feat_prop1 和 feat_prop2 是将融合后的特征分割成两部分，并分别通过卷积层和激活函数处理。
    feat_prop 是将处理后的特征进行融合，得到传播的特征。
    feat_prop 通过动态像素卷积层处理。
    feat_prop 通过2D残差块处理。
    forward_list 将处理后的特征添加到列表中。
    conv3d_feature 将列表中的特征堆叠起来，形成最终的输出特征图。
    '''
