import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple


class CA_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CA_Block, self).__init__()

        self.conv_1x1 = nn.Conv2d(
            in_channels=channel,
            out_channels=channel // reduction,
            kernel_size=1,
            stride=1,
            bias=False,
        )

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)

        self.F_h = nn.Conv2d(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.F_w = nn.Conv2d(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            bias=False,
        )

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        _, _, h, w = x.size()

        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim=2, keepdim=True)

        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out


def _get_act_fn(act_name, inplace=True):
    if act_name == "relu":
        return nn.ReLU(inplace=inplace)
    elif act_name == "leaklyrelu":
        return nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
    elif act_name == "gelu":
        return nn.GELU()
    else:
        raise NotImplementedError


class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        act_name="relu",
        is_transposed=False,
    ):
        """
        Convolution-BatchNormalization-ActivationLayer

        :param in_planes:
        :param out_planes:
        :param kernel_size:
        :param stride:
        :param padding:
        :param dilation:
        :param groups:
        :param bias:
        :param act_name: None denote it doesn't use the activation layer.
        :param is_transposed: True -> nn.ConvTranspose2d, False -> nn.Conv2d
        """
        super().__init__()
        if is_transposed:
            conv_module = nn.ConvTranspose2d
        else:
            conv_module = nn.Conv2d
        self.add_module(
            name="conv",
            module=conv_module(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=to_2tuple(stride),
                padding=to_2tuple(padding),
                dilation=to_2tuple(dilation),
                groups=groups,
                bias=bias,
            ),
        )
        self.add_module(name="bn", module=nn.BatchNorm2d(out_channels))
        if act_name is not None:
            self.add_module(name=act_name, module=_get_act_fn(act_name=act_name))


class HMU(nn.Module):
    def __init__(self, in_c, num_groups=4, hidden_dim=None):
        super().__init__()
        self.num_groups = num_groups

        hidden_dim = hidden_dim or in_c // 2
        expand_dim = hidden_dim * num_groups
        self.expand_conv = ConvBNReLU(in_c, expand_dim, 1)
        self.gate_genator = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(num_groups * hidden_dim, hidden_dim, 1),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim, num_groups * hidden_dim, 1),
            nn.Softmax(dim=1),
        )

        self.interact = nn.ModuleDict()
        self.interact["0"] = ConvBNReLU(hidden_dim, 3 * hidden_dim, 3, 1, 1)
        for group_id in range(1, num_groups - 1):
            self.interact[str(group_id)] = ConvBNReLU(
                2 * hidden_dim, 3 * hidden_dim, 3, 1, 1
            )
        self.interact[str(num_groups - 1)] = ConvBNReLU(
            2 * hidden_dim, 2 * hidden_dim, 3, 1, 1
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(num_groups * hidden_dim, in_c, 3, 1, 1),
            nn.BatchNorm2d(in_c),
            nn.ReLU(True),
        )

    def forward(self, x):
        xs = self.expand_conv(x).chunk(self.num_groups, dim=1)

        outs = []

        branch_out = self.interact["0"](xs[0])
        outs.append(branch_out.chunk(3, dim=1))

        for group_id in range(1, self.num_groups - 1):
            branch_out = self.interact[str(group_id)](
                torch.cat([xs[group_id], outs[group_id - 1][1]], dim=1)
            )
            outs.append(branch_out.chunk(3, dim=1))

        group_id = self.num_groups - 1
        branch_out = self.interact[str(group_id)](
            torch.cat([xs[group_id], outs[group_id - 1][1]], dim=1)
        )
        outs.append(branch_out.chunk(2, dim=1))

        out = torch.cat([o[0] for o in outs], dim=1)
        gate = self.gate_genator(torch.cat([o[-1] for o in outs], dim=1))
        out = self.fuse(out * gate) + x
        return out


class FRB(nn.Module):
    def __init__(self, in_channels):
        super(FRB, self).__init__()
        self.in_channels = in_channels
        self.left = nn.Sequential(
            ConvBNReLU(
                in_channels=self.in_channels,
                out_channels=self.in_channels // 2,
                kernel_size=1,
                stride=1,
            ),
            nn.ConvTranspose2d(
                in_channels=self.in_channels // 2,
                out_channels=self.in_channels // 2,
                kernel_size=2,
                stride=2,
            ),
            ConvBNReLU(
                in_channels=self.in_channels // 2,
                out_channels=self.in_channels // 2,
                kernel_size=2,
                stride=2,
            ),
            ConvBNReLU(
                in_channels=self.in_channels // 2,
                out_channels=self.in_channels,
                kernel_size=1,
                stride=1,
            ),
        )
        self.right = nn.Sequential(
            ConvBNReLU(
                in_channels=self.in_channels,
                out_channels=self.in_channels // 2,
                kernel_size=1,
                stride=1,
            ),
            ConvBNReLU(
                in_channels=self.in_channels // 2,
                out_channels=self.in_channels // 2,
                kernel_size=2,
                stride=2,
            ),
            nn.ConvTranspose2d(
                in_channels=self.in_channels // 2,
                out_channels=self.in_channels // 2,
                kernel_size=2,
                stride=2,
            ),
            ConvBNReLU(
                in_channels=self.in_channels // 2,
                out_channels=self.in_channels,
                kernel_size=1,
                stride=1,
            ),
        )

    def forward(self, input):
        l = self.left(input)
        r = self.right(input)
        return l + r


class OAA(nn.Module):
    def __init__(
        self,
        cur_in_channels=64,
        low_in_channels=32,
        out_channels=16,
        cur_scale=2,
        low_scale=1,
    ):
        super(OAA, self).__init__()
        self.cur_in_channels = cur_in_channels
        self.cur_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=cur_in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
        )
        self.low_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=low_in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
        )

        self.cur_scale = cur_scale
        self.low_scale = low_scale

        self.out_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels * 2,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
        )

    def forward(self, x_cur, x_low):
        x_cur = self.cur_conv(x_cur)
        # bicubic bilinear nearest
        x_cur = F.interpolate(
            x_cur, scale_factor=self.cur_scale, mode="bilinear", align_corners=True
        )

        x_low = self.low_conv(x_low)
        x_low = F.interpolate(
            x_low, scale_factor=self.low_scale, mode="bilinear", align_corners=True
        )
        x = torch.cat((x_cur, x_low), dim=1)
        x = self.out_conv(x)
        return x


class BasicConv2d(nn.Module):
    def __init__(
        self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1
    ):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3),
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5),
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7),
        )
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):

        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class NeighborConnectionDecoder(nn.Module):
    def __init__(self, channel):
        super(NeighborConnectionDecoder, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.cr1 = nn.Conv2d(256, channel, 1)
        self.cr2 = nn.Conv2d(128, channel, 1)
        self.conv_upsample1 = nn.Conv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = nn.Conv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = nn.Conv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = nn.Conv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = nn.Conv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = nn.Conv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = nn.Conv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = ConvBNReLU(3 * channel, 3 * channel, 3, padding=1)
        self.conv5 = ConvBNReLU(3 * channel, channel, 1)

    def forward(self, x1, x2, x3):
        x1 = self.cr1(x1)
        x2 = self.cr2(x2)
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = (
            self.conv_upsample2(self.upsample(x2_1))
            * self.conv_upsample3(self.upsample(x2))
            * x3
        )

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x


class IFA(nn.Module):
    def __init__(self, channel):
        super(IFA, self).__init__()
        self.down = nn.Conv2d(channel, channel // 2, 1)
        self.conv3 = nn.Conv2d(channel // 2, channel // 2, 3, 1, 1)
        self.conv5 = nn.Sequential(
            nn.Conv2d(channel // 2, channel // 2, 3, 1, 1),
            nn.Conv2d(channel // 2, channel // 2, 3, 1, 1),
        )
        self.conv353 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.conv355 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.Conv2d(channel, channel, 3, 1, 1),
        )
        self.res = nn.Conv2d(channel, channel, 3, 1, 1)
        self.conca = ConvBNReLU(channel * 2, channel, 3, 1, 1)

    def forward(self, x):
        x_ = x
        x = self.down(x)
        fk3 = self.conv3(x)
        fk5 = self.conv5(x)
        fk35 = torch.concat((fk3, fk5), 1)
        fk35 = self.conv353(fk35) + self.conv355(fk35)
        fconcat = torch.concat((fk3, fk5), 1)
        fconcat = torch.cat((fconcat, fk35), 1)
        fka = self.res(x_) + self.conca(fconcat)
        return fka
