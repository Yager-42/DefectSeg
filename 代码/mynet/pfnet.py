"""
 @Time    : 2021/7/6 14:23
 @Author  : Haiyang Mei
 @E-mail  : mhy666@mail.dlut.edu.cn

 @Project : CVPR2021_PFNet
 @File    : PFNet.py
 @Function: Focus and Exploration Network

"""

from timm.models.pvt_v2 import _cfg
from mynet.backbone.pvt.pvt_v2 import pvt_v2_b2
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone.resnet import resnet as resnet
from config import UNetConfig
from .model_utils import OAA, FRB, NeighborConnectionDecoder

cfg = UNetConfig()


###################################################################
# ################## Channel Attention Block ######################
###################################################################
class CA_Block(nn.Module):
    def __init__(self, in_dim):
        super(CA_Block, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        inputs :
            x : input feature maps (B X C X H X W)
        returns :
            out : channel attentive features
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


###################################################################
# ################## Spatial Attention Block ######################
###################################################################
class SA_Block(nn.Module):
    def __init__(self, in_dim):
        super(SA_Block, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1
        )
        self.key_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1
        )
        self.value_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1
        )
        self.gamma = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        inputs :
            x : input feature maps (B X C X H X W)
        returns :
            out : spatial attentive features
        """
        m_batchsize, C, height, width = x.size()
        proj_query = (
            self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        )
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


###################################################################
# ################## Context Exploration Block ####################
###################################################################
class Context_Exploration_Block(nn.Module):
    def __init__(self, input_channels):
        super(Context_Exploration_Block, self).__init__()
        self.input_channels = input_channels
        self.channels_single = int(input_channels / 4)

        self.p1_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single),
            nn.ReLU(),
        )
        self.p2_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single),
            nn.ReLU(),
        )
        self.p3_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single),
            nn.ReLU(),
        )
        self.p4_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single),
            nn.ReLU(),
        )

        self.p1 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single),
            nn.ReLU(),
        )
        self.p1_dc = nn.Sequential(
            nn.Conv2d(
                self.channels_single,
                self.channels_single,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
            ),
            nn.BatchNorm2d(self.channels_single),
            nn.ReLU(),
        )

        self.p2 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 3, 1, 1),
            nn.BatchNorm2d(self.channels_single),
            nn.ReLU(),
        )
        self.p2_dc = nn.Sequential(
            nn.Conv2d(
                self.channels_single,
                self.channels_single,
                kernel_size=3,
                stride=1,
                padding=2,
                dilation=2,
            ),
            nn.BatchNorm2d(self.channels_single),
            nn.ReLU(),
        )

        self.p3 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 5, 1, 2),
            nn.BatchNorm2d(self.channels_single),
            nn.ReLU(),
        )
        self.p3_dc = nn.Sequential(
            nn.Conv2d(
                self.channels_single,
                self.channels_single,
                kernel_size=3,
                stride=1,
                padding=4,
                dilation=4,
            ),
            nn.BatchNorm2d(self.channels_single),
            nn.ReLU(),
        )

        self.p4 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 7, 1, 3),
            nn.BatchNorm2d(self.channels_single),
            nn.ReLU(),
        )
        self.p4_dc = nn.Sequential(
            nn.Conv2d(
                self.channels_single,
                self.channels_single,
                kernel_size=3,
                stride=1,
                padding=8,
                dilation=8,
            ),
            nn.BatchNorm2d(self.channels_single),
            nn.ReLU(),
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(self.input_channels, self.input_channels, 1, 1, 0),
            nn.BatchNorm2d(self.input_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        p1_input = self.p1_channel_reduction(x)
        p1 = self.p1(p1_input)
        p1_dc = self.p1_dc(p1)

        p2_input = self.p2_channel_reduction(x) + p1_dc
        p2 = self.p2(p2_input)
        p2_dc = self.p2_dc(p2)

        p3_input = self.p3_channel_reduction(x) + p2_dc
        p3 = self.p3(p3_input)
        p3_dc = self.p3_dc(p3)

        p4_input = self.p4_channel_reduction(x) + p3_dc
        p4 = self.p4(p4_input)
        p4_dc = self.p4_dc(p4)

        ce = self.fusion(torch.cat((p1_dc, p2_dc, p3_dc, p4_dc), 1))

        return ce


###################################################################
# ##################### Positioning Module ########################
###################################################################
class Positioning(nn.Module):
    def __init__(self, channel):
        super(Positioning, self).__init__()
        self.channel = channel
        self.cab = CA_Block(self.channel)
        self.sab = SA_Block(self.channel)
        self.map = nn.Conv2d(self.channel, 1, 7, 1, 3)

    def forward(self, x):
        cab = self.cab(x)
        sab = self.sab(cab)
        map = self.map(sab)

        return sab, map


###################################################################
# ######################## Focus Module ###########################
###################################################################
class Focus(nn.Module):
    def __init__(self, channel1, channel2, is_last=False):
        super(Focus, self).__init__()
        self.channel1 = channel1
        self.channel2 = channel2

        self.up = nn.Sequential(
            nn.Conv2d(self.channel2, self.channel1, 7, 1, 3),
            nn.BatchNorm2d(self.channel1),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

        self.input_map = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2), nn.Sigmoid()
        )
        if not is_last:
            self.output_map = nn.Conv2d(self.channel1, 1, 7, 1, 3)
        else:
            # 多分类
            self.output_map = nn.Sequential(
                nn.Conv2d(self.channel1, cfg.n_classes, 7, 1, 3)
            )

        self.fp = Context_Exploration_Block(self.channel1)
        self.fn = Context_Exploration_Block(self.channel1)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.bn1 = nn.BatchNorm2d(self.channel1)
        self.relu1 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(self.channel1)
        self.relu2 = nn.ReLU()

    def forward(self, x, y, in_map):
        # x; current-level features
        # y: higher-level features
        # in_map: higher-level prediction

        up = self.up(y)

        input_map = self.input_map(in_map)
        f_feature = x * input_map
        b_feature = x * (1 - input_map)

        fp = self.fp(f_feature)
        fn = self.fn(b_feature)

        refine1 = up - (self.alpha * fp)
        refine1 = self.bn1(refine1)
        refine1 = self.relu1(refine1)

        refine2 = refine1 + (self.beta * fn)
        refine2 = self.bn2(refine2)
        refine2 = self.relu2(refine2)

        output_map = self.output_map(refine2)

        return refine2, output_map


###################################################################
# ######################## SqueezeAttentionBlock ###########################
###################################################################
class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class SqueezeAttentionBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(SqueezeAttentionBlock, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv = conv_block(ch_in, ch_out)
        self.conv_atten = conv_block(ch_in, ch_out)
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x):
        # print(x.shape)
        x_res = self.conv(x)
        # print(x_res.shape)
        y = self.avg_pool(x)
        # print(y.shape)
        y = self.conv_atten(y)
        # print(y.shape)
        y = self.upsample(y)
        # print(y.shape, x_res.shape)
        return (y * x_res) + y


###################################################################
# ########################## NETWORK ##############################
###################################################################


def get_timm_pretrained_model():
    # file为本地文件路径
    config = _cfg(url="", file="mynet/pretrained_backbone/pvt_b2.bin")
    model = timm.create_model(
        "pvt_v2_b2", pretrained=True, features_only=True, pretrained_cfg=config
    )
    return model


def get_pvt():
    backbone = pvt_v2_b2()
    path = "mynet/backbone/pvt/pvt_v2_b2.pth"
    save_model = torch.load(path)
    model_dict = backbone.state_dict()
    state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    backbone.load_state_dict(model_dict)
    return backbone


class PFNet(nn.Module):
    def __init__(
        self, cfg, backbone_path="mynet/backbone/resnet/resnet50-19c8e357.pth"
    ):
        super(PFNet, self).__init__()
        # params

        # backbone
        # resnet50 = resnet.resnet50(backbone_path)
        # self.layer0 = nn.Sequential(
        #    resnet50.conv1, resnet50.bn1, resnet50.relu)
        # self.layer1 = nn.Sequential(resnet50.maxpool, resnet50.layer1)
        # self.layer2 = resnet50.layer2
        # self.layer3 = resnet50.layer3
        # self.layer4 = resnet50.layer4
        # channel reduction
        # self.cr4 = nn.Sequential(
        #    nn.Conv2d(2048, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU())
        # self.cr3 = nn.Sequential(
        #    nn.Conv2d(1024, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU())
        # self.cr2 = nn.Sequential(
        #    nn.Conv2d(512, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU())
        # self.cr1 = nn.Sequential(
        #    nn.Conv2d(256, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())

        self.model = get_pvt()
        # channel reduction
        self.cr4 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU()
        )
        self.cr3 = nn.Sequential(
            nn.Conv2d(320, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU()
        )
        self.cr2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU()
        )
        self.cr1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU()
        )

        self.oaa1 = OAA(
            cur_in_channels=64,
            low_in_channels=128,
            out_channels=64,
            cur_scale=1,
            low_scale=2,
        )
        self.oaa2 = OAA(
            cur_in_channels=128,
            low_in_channels=256,
            out_channels=128,
            cur_scale=1,
            low_scale=2,
        )
        self.oaa3 = OAA(
            cur_in_channels=256,
            low_in_channels=512,
            out_channels=256,
            cur_scale=1,
            low_scale=2,
        )
        self.oaa0 = OAA(
            cur_in_channels=64,
            low_in_channels=512,
            out_channels=64,
            cur_scale=1,
            low_scale=8,
        )

        # positioning
        self.positioning = Positioning(512)
        # sa1和predict1做attn
        self.attn = nn.Sequential(SqueezeAttentionBlock(cfg.n_classes, cfg.n_classes))

        # focus
        self.focus3 = Focus(256, 512)
        self.focus2 = Focus(128, 256)
        self.focus1 = Focus(64, 128, is_last=True)

        # 多分类
        self.mul4 = nn.Sequential(
            nn.Conv2d(512, cfg.n_classes, 7, 1, 3),
            nn.BatchNorm2d(cfg.n_classes),
            nn.ReLU(inplace=True),
        )
        self.mul3 = nn.Sequential(
            nn.Conv2d(256, cfg.n_classes, 7, 1, 3),
            nn.BatchNorm2d(cfg.n_classes),
            nn.ReLU(inplace=True),
        )
        self.mul2 = nn.Sequential(
            nn.Conv2d(128, cfg.n_classes, 7, 1, 3),
            nn.BatchNorm2d(cfg.n_classes),
            nn.ReLU(inplace=True),
        )

        self.edge1 = nn.Sequential(
            nn.Conv2d(cfg.n_classes, 1, 7, 1, 3),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
        self.edge2 = nn.Sequential(
            nn.Conv2d(cfg.n_classes, 1, 7, 1, 3),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
        self.edge3 = nn.Sequential(
            nn.Conv2d(cfg.n_classes, 1, 7, 1, 3),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
        self.edge4 = nn.Sequential(
            nn.Conv2d(cfg.n_classes, 1, 7, 1, 3),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )

        # 做块聚集
        self.sa4 = SqueezeAttentionBlock(ch_in=cfg.n_classes, ch_out=cfg.n_classes)
        self.sa3 = SqueezeAttentionBlock(ch_in=cfg.n_classes, ch_out=cfg.n_classes)
        self.sa2 = SqueezeAttentionBlock(ch_in=cfg.n_classes, ch_out=cfg.n_classes)

        self.sa1 = SqueezeAttentionBlock(ch_in=cfg.n_classes, ch_out=cfg.n_classes)

        # 聚合所有块
        # self.class_wise_mask = nn.Sequential(nn.Conv2d(cfg.n_classes * 4, cfg.n_classes, kernel_size=1),
        #                                     nn.BatchNorm2d(cfg.n_classes),
        #                                     nn.ReLU(inplace=True))

        # mask类别
        self.categorical1 = nn.Sequential(
            nn.BatchNorm2d(cfg.n_classes),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(cfg.n_classes, cfg.n_classes, 1),
        )
        self.categorical2 = nn.Sequential(
            nn.BatchNorm2d(cfg.n_classes),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(cfg.n_classes, cfg.n_classes, 1),
        )
        self.categorical3 = nn.Sequential(
            nn.BatchNorm2d(cfg.n_classes),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(cfg.n_classes, cfg.n_classes, 1),
        )
        self.categorical4 = nn.Sequential(
            nn.BatchNorm2d(cfg.n_classes),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(cfg.n_classes, cfg.n_classes, 1),
        )

        self.out1 = nn.Sequential(
            nn.BatchNorm2d(cfg.n_classes),
            nn.ReLU(inplace=True),
            nn.Conv2d(cfg.n_classes, cfg.n_classes, 3, 1, 1),
        )
        self.out2 = nn.Sequential(
            nn.BatchNorm2d(cfg.n_classes),
            nn.ReLU(inplace=True),
            nn.Conv2d(cfg.n_classes, cfg.n_classes, 3, 1, 1),
        )
        self.out3 = nn.Sequential(
            nn.BatchNorm2d(cfg.n_classes),
            nn.ReLU(inplace=True),
            nn.Conv2d(cfg.n_classes, cfg.n_classes, 3, 1, 1),
        )
        self.out4 = nn.Sequential(
            nn.BatchNorm2d(cfg.n_classes),
            nn.ReLU(inplace=True),
            nn.Conv2d(cfg.n_classes, cfg.n_classes, 3, 1, 1),
        )

        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x):
        # x: [batch_size, channel=3, h, w]
        # layer0 = self.layer0(x)  # [-1, 64, h/2, w/2]
        # layer1 = self.layer1(layer0)  # [-1, 256, h/4, w/4]
        # layer2 = self.layer2(layer1)  # [-1, 512, h/8, w/8]
        # layer3 = self.layer3(layer2)  # [-1, 1024, h/16, w/16]
        # layer4 = self.layer4(layer3)  # [-1, 2048, h/32, w/32]
        layers = self.model(x)
        layer1 = layers[0]
        layer2 = layers[1]
        layer3 = layers[2]
        layer4 = layers[3]

        # channel reduction
        cr4 = self.cr4(layer4)
        cr3 = self.cr3(layer3)
        cr2 = self.cr2(layer2)
        cr1 = self.cr1(layer1)

        cr3 = self.oaa3(cr3, cr4)
        cr2 = self.oaa2(cr2, cr3)
        cr1 = self.oaa1(cr1, cr2)
        cr1 = self.oaa0(cr1, cr4)

        # positioning
        positioning, predict4 = self.positioning(cr4)

        # focus
        focus3, predict3 = self.focus3(cr3, positioning, predict4)
        focus2, predict2 = self.focus2(cr2, focus3, predict3)
        focus1, predict1 = self.focus1(cr1, focus2, predict2)

        sa4 = self.mul4(positioning)
        sa3 = self.mul3(focus3)
        sa2 = self.mul2(focus2)

        # 根据每个阶段的输出得到块聚合
        sa1 = self.sa1(predict1)
        predict1 = predict1 + sa1
        predict1 = self.out1(predict1)

        sa2 = self.sa2(sa2)
        predict2 = predict2 + sa2
        predict2 = self.out2(predict2)

        sa3 = self.sa3(sa3)
        predict3 = predict3 + sa3
        predict3 = self.out3(predict3)

        sa4 = self.sa4(sa4)
        predict4 = predict4 + sa4
        predict4 = self.out4(predict4)

        edge1 = self.edge1(predict1)
        edge2 = self.edge2(predict2)
        edge3 = self.edge3(predict3)
        edge4 = self.edge4(predict4)

        categorical1 = self.categorical1(predict1).squeeze(-1).squeeze(-1)
        categorical2 = self.categorical2(predict2).squeeze(-1).squeeze(-1)
        categorical3 = self.categorical3(predict3).squeeze(-1).squeeze(-1)
        categorical4 = self.categorical4(predict4).squeeze(-1).squeeze(-1)

        # 放到统一尺寸
        predict4 = F.interpolate(
            predict4, size=x.size()[2:], mode="bilinear", align_corners=True
        )
        predict3 = F.interpolate(
            predict3, size=x.size()[2:], mode="bilinear", align_corners=True
        )
        predict2 = F.interpolate(
            predict2, size=x.size()[2:], mode="bilinear", align_corners=True
        )
        predict1 = F.interpolate(
            predict1, size=x.size()[2:], mode="bilinear", align_corners=True
        )

        edge4 = F.interpolate(
            edge4, size=x.size()[2:], mode="bilinear", align_corners=True
        )
        edge3 = F.interpolate(
            edge3, size=x.size()[2:], mode="bilinear", align_corners=True
        )
        edge2 = F.interpolate(
            edge2, size=x.size()[2:], mode="bilinear", align_corners=True
        )
        edge1 = F.interpolate(
            edge1, size=x.size()[2:], mode="bilinear", align_corners=True
        )

        if self.training:
            return (
                predict4,
                predict3,
                predict2,
                predict1,
                categorical1,
                categorical2,
                categorical3,
                categorical4,
                edge1,
                edge2,
                edge3,
                edge4,
            )

        return (
            predict4,
            predict3,
            predict2,
            predict1,
            categorical1,
            categorical2,
            categorical3,
            categorical4,
        )
