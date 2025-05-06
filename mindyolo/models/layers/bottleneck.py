from mindspore import nn, ops,mint, Tensor
import math
import mindspore.numpy as mnp
from ..layers.utils import meshgrid
from .conv import ConvNormAct, DWConvNormAct
import numpy as np
import pdb

class Bottleneck(nn.Cell):
    # Standard bottleneck
    def __init__(
        self, c1, c2, shortcut=True, k=(1, 3), g=(1, 1), e=0.5, act=True, momentum=0.97, eps=1e-3, sync_bn=False
    ):  # ch_in, ch_out, shortcut, kernels, group, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.conv1 = ConvNormAct(c1, c_, k[0], 1, g=g[0], act=act, momentum=momentum, eps=eps, sync_bn=sync_bn)
        self.conv2 = ConvNormAct(c_, c2, k[1], 1, g=g[1], act=act, momentum=momentum, eps=eps, sync_bn=sync_bn)
        self.add = shortcut and c1 == c2

    def construct(self, x):
        if self.add:
            out = x + self.conv2(self.conv1(x))
        else:
            out = self.conv2(self.conv1(x))
        return out

    
class Residualblock(nn.Cell):
    def __init__(
        self, c1, c2, k=(1, 3), g=(1, 1), act=True, momentum=0.97, eps=1e-3, sync_bn=False
    ):  # ch_in, ch_out, kernels, group, expand
        super().__init__()
        self.conv1 = ConvNormAct(c1, c2, k[0], 1, g=g[0], act=act, momentum=momentum, eps=eps, sync_bn=sync_bn)
        self.conv2 = ConvNormAct(c2, c2, k[1], 1, g=g[1], act=act, momentum=momentum, eps=eps, sync_bn=sync_bn)

    def construct(self, x):
        out = x + self.conv2(self.conv1(x))
        return out


class C3(nn.Cell):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5, momentum=0.97, eps=1e-3, sync_bn=False):
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.conv1 = ConvNormAct(c1, c_, 1, 1, momentum=momentum, eps=eps, sync_bn=sync_bn)
        self.conv2 = ConvNormAct(c1, c_, 1, 1, momentum=momentum, eps=eps, sync_bn=sync_bn)
        self.conv3 = ConvNormAct(2 * c_, c2, 1, momentum=momentum, eps=eps, sync_bn=sync_bn)  # act=FReLU(c2)
        self.m = nn.SequentialCell(
            [
                Bottleneck(c_, c_, shortcut, k=(1, 3), e=1.0, momentum=momentum, eps=eps, sync_bn=sync_bn)
                for _ in range(n)
            ]
        )
        self.concat = ops.Concat(axis=1)

    def construct(self, x):
        c1 = self.conv1(x)
        c2 = self.m(c1)
        c3 = self.conv2(x)
        c4 = self.concat((c2, c3))
        c5 = self.conv3(c4)

        return c5


class C2f(nn.Cell):
    # CSP Bottleneck with 2 convolutions
    def __init__(
        self, c1, c2, n=1, shortcut=False, g=1, e=0.5, momentum=0.97, eps=1e-3, sync_bn=False
    ):  # ch_in, ch_out, number, shortcut, group, expansion
        super().__init__()
        _c = int(c2 * e)  # hidden channels
        self.cv1 = ConvNormAct(c1, 2 * _c, 1, 1, momentum=momentum, eps=eps, sync_bn=sync_bn)
        self.cv2 = ConvNormAct(
            (2 + n) * _c, c2, 1, momentum=momentum, eps=eps, sync_bn=sync_bn
        )  # optional act=FReLU(c2)
        self.m = nn.CellList(
            [
                Bottleneck(_c, _c, shortcut, k=(3, 3), g=(1, g), e=1.0, momentum=momentum, eps=eps, sync_bn=sync_bn)
                for _ in range(n)
            ]
        )

    def construct(self, x):
        y = ()
        x = self.cv1(x)
        _c = x.shape[1] // 2
        x_tuple = ops.split(x, axis=1, split_size_or_sections=_c)
        y += x_tuple
        for i in range(len(self.m)):
            m = self.m[i]
            out = m(y[-1])
            y += (out,)

        return self.cv2(ops.concat(y, axis=1))


class DWBottleneck(nn.Cell):
    # depthwise bottleneck used in yolox nano scale
    def __init__(
        self, c1, c2, shortcut=True, k=(1, 3), e=0.5, act=True, momentum=0.97, eps=1e-3, sync_bn=False
    ):  # ch_in, ch_out, shortcut, group, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.conv1 = ConvNormAct(c1, c_, k[0], 1, act=True, momentum=momentum, eps=eps, sync_bn=sync_bn)
        self.conv2 = DWConvNormAct(c_, c2, k[1], 1, act=True, momentum=momentum, eps=eps, sync_bn=sync_bn)
        self.add = shortcut and c1 == c2

    def construct(self, x):
        if self.add:
            out = x + self.conv2(self.conv1(x))
        else:
            out = self.conv2(self.conv1(x))
        return out


class DWC3(nn.Cell):
    # depthwise DwC3 used in yolox nano scale, similar as C3
    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5, momentum=0.97, eps=1e-3, sync_bn=False):
        super(DWC3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.conv1 = ConvNormAct(c1, c_, 1, 1, momentum=momentum, eps=eps, sync_bn=sync_bn)
        self.conv2 = ConvNormAct(c1, c_, 1, 1, momentum=momentum, eps=eps, sync_bn=sync_bn)
        self.conv3 = ConvNormAct(2 * c_, c2, 1, momentum=momentum, eps=eps, sync_bn=sync_bn)  # act=FReLU(c2)
        self.m = nn.SequentialCell(
            [
                DWBottleneck(c_, c_, shortcut, k=(1, 3), e=1.0, momentum=momentum, eps=eps, sync_bn=sync_bn)
                for _ in range(n)
            ]
        )
        self.concat = ops.Concat(axis=1)

    def construct(self, x):
        c1 = self.conv1(x)
        c2 = self.m(c1)
        c3 = self.conv2(x)
        c4 = self.concat((c2, c3))
        c5 = self.conv3(c4)

        return c5

#=================================
import mindspore as ms


decay = 0.25  # 0.25 # decay constants
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class MultiSpike4(nn.Cell):
    def __init__(self):
        super(MultiSpike4, self).__init__()

    def construct(self, x):
        out = ops.round(ops.clamp(x, min=0, max=4))
        return out

    def bprop(self, x, out, dout): #x是前传的输入，out是前传的结果，dout是梯度
        dout[x < 0] = 0
        dout[x > 4] = 0
        return  dout

class mem_update(nn.Cell):
    def __init__(self, act=False):
        super(mem_update, self).__init__()
        self.act = act
        self.qtrick = MultiSpike4()

    def construct(self, x):
        spike = ops.zeros_like(x[0])
        output = ops.zeros_like(x)
        mem = 0
        time_window = x.shape[0]

        for i in range(time_window):
            if i >= 1:
                mem = (mem - ops.stop_gradient(spike)) * decay + x[i]
            else:
                mem = x[i]
            spike = self.qtrick(mem)
            output[i] = spike

        return output

class BNAndPadLayer(nn.Cell):
    def __init__(
            self,
            pad_pixels,
            num_features,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
    ):
        super(BNAndPadLayer, self).__init__()
        self.affine = affine
        self.bn = nn.BatchNorm2d(
            num_features, eps, momentum, affine
        )
        self.pad_pixels = pad_pixels


    def construct(self, input):
        print("===============================input:", input)
        output = self.bn(input)
        if self.pad_pixels > 0:
            if self.affine:

                pad_values = self.bn.beta- self.bn.moving_mean* self.bn.gamma/ (ops.sqrt( abs(self.bn.moving_variance) ) + self.bn.eps)
                # import pdb
                # pdb.set_trace()
                print("]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]:",pad_values)

            else:
                pad_values = -self.bn.moving_mean / ops.sqrt(
                    self.bn.moving_variance + self.bn.eps
                )
            pad_op = ops.Pad(
                paddings=((0, 0), (0, 0), (self.pad_pixels, self.pad_pixels), (self.pad_pixels, self.pad_pixels)))
            output = pad_op(output)



            pad_values = pad_values.view(1, -1, 1, 1)

            output[:, :, 0: self.pad_pixels, :] = pad_values
            output[:, :, -self.pad_pixels:, :] = pad_values
            output[:, :, :, 0: self.pad_pixels] = pad_values
            output[:, :, :, -self.pad_pixels:] = pad_values

            # output[:, :, 0: self.pad_pixels, :] = pad_values[:, :, 0: self.pad_pixels, :]
            # output[:, :, -self.pad_pixels:, :] = pad_values[:, :, -self.pad_pixels:, :]
            # output[:, :, :, 0: self.pad_pixels] = pad_values[:, :, :, 0: self.pad_pixels]
            # output[:, :, :, -self.pad_pixels:] = pad_values[:, :, :, -self.pad_pixels:]

        return output


# class BNAndPadLayer(nn.Cell):
#     def __init__(self, pad_pixels, num_features, eps=1e-5, momentum=0.1, affine=True):
#         super(BNAndPadLayer, self).__init__()
#         self.affine = affine
#         self.bn = nn.BatchNorm2d(num_features, eps, momentum, use_batch_statistics=True)
#         self.pad_pixels = pad_pixels
#
#     def construct(self, input):
#         output = self.bn(input)
#         if self.pad_pixels > 0:
#             if self.affine:
#                 pad_values = (
#                         ops.stop_gradient(self.bn.beta)
#                         - self.bn.moving_mean
#                         * ops.stop_gradient(self.bn.gamma)
#                         / ops.sqrt(self.bn.moving_variance + self.bn.eps)
#                 )
#             else:
#                 pad_values = -self.bn.moving_mean / ops.sqrt(
#                     self.bn.moving_variance + self.bn.eps
#                 )
#             pad_op = ops.Pad(
#                 paddings=((0, 0), (0, 0), (self.pad_pixels, self.pad_pixels), (self.pad_pixels, self.pad_pixels)))
#             output = pad_op(output)
#             pad_values = pad_values.view(1, -1, 1, 1)
#
#             output[:, :, 0:self.pad_pixels, :] = pad_values
#             output[:, :, -self.pad_pixels:, :] = pad_values
#             output[:, :, :, 0:self.pad_pixels] = pad_values
#             output[:, :, :, -self.pad_pixels:] = pad_values
#         return output


class RepConv(nn.Cell):
    def __init__(self, in_channel, out_channel, kernel_size=3, has_bias=False, group=1):
        super(RepConv, self).__init__()
        padding = int((kernel_size - 1) / 2)
        conv1x1 = nn.Conv2d(in_channel, in_channel, 1, has_bias=has_bias, group=group)
        bn = BNAndPadLayer(pad_pixels=padding, num_features=in_channel)
        conv3x3 = nn.SequentialCell([
            nn.Conv2d(in_channel, in_channel, kernel_size, group=in_channel, has_bias=has_bias,padding=0,pad_mode="pad"),
            nn.Conv2d(in_channel, out_channel, 1, group=group, has_bias=has_bias),
            nn.BatchNorm2d(out_channel),
        ])
        self.body = nn.SequentialCell([conv1x1, bn, conv3x3])

    def construct(self, x):
        return self.body(x)


class SepRepConv(nn.Cell):
    def __init__(self, in_channel, out_channel, kernel_size=3, has_bias=False, group=1):
        super(SepRepConv, self).__init__()
        padding = int((kernel_size - 1) / 2)
        self.bn = BNAndPadLayer(pad_pixels=padding, num_features=in_channel)
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size, group=in_channel, has_bias=has_bias,padding=0,pad_mode="pad")
        self.conv2 =  nn.Conv2d(in_channel, out_channel, 1, group=group, has_bias=has_bias)


        # conv3x3 = nn.SequentialCell([
        #     nn.Conv2d(in_channel, in_channel, kernel_size, group=in_channel, has_bias=has_bias,padding=0,pad_mode="pad"),
        #     nn.Conv2d(in_channel, out_channel, 1, group=group, has_bias=has_bias),
        # ])
        # self.body = nn.SequentialCell([bn, conv3x3])

    def construct(self, x):

        x = self.bn(x)
        x = self.conv1(x)
        x = self.conv2(x)

        return x #111111111234此行报错


class SepConv(nn.Cell):
    def __init__(self, dim, expansion_ratio=2, kernel_size=3):
        super(SepConv, self).__init__()
        padding = int((kernel_size - 1) / 2)
        med_channels = int(expansion_ratio * dim)
        self.pwconv1 = nn.Conv2d(dim, med_channels, kernel_size=1, stride=1, pad_mode="pad")
        self.dwconv2 = nn.Conv2d(med_channels, med_channels, kernel_size=kernel_size,
                                 padding=padding, group=med_channels, pad_mode="pad")
        self.pwconv3 = SepRepConv(med_channels, dim)
        self.bn1 = nn.BatchNorm2d(med_channels)
        self.bn2 = nn.BatchNorm2d(med_channels)
        self.bn3 = nn.BatchNorm2d(dim)

    def construct(self, x):
        T, B, C, H, W = x.shape
        x = self.bn1(self.pwconv1(ops.flatten(x, start_dim=0, end_dim=1))).view(T, B, -1, H, W)
        x = self.bn2(self.dwconv2(ops.flatten(x, start_dim=0, end_dim=1))).view(T, B, -1, H, W)
        x = self.bn3(self.pwconv3(ops.flatten(x, start_dim=0, end_dim=1))).view(T, B, -1, H, W)
        return x


class MS_ConvBlock(nn.Cell):
    def __init__(self, input_dim, mlp_ratio=4., sep_kernel_size=7):
        super(MS_ConvBlock, self).__init__()
        self.Conv = SepConv(dim=input_dim, kernel_size=sep_kernel_size)
        self.mlp_ratio = mlp_ratio
        self.conv1 = RepConv(input_dim, int(input_dim * mlp_ratio))
        self.bn1 = nn.BatchNorm2d(int(input_dim * mlp_ratio))
        self.conv2 = RepConv(int(input_dim * mlp_ratio), input_dim)
        self.bn2 = nn.BatchNorm2d(input_dim)

    def construct(self, x):
        T, B, C, H, W = x.shape
        x = self.Conv(x) + x
        x_feat = x
        x = self.bn1(self.conv1(ops.flatten(x, start_dim=0, end_dim=1))).view(T, B, int(self.mlp_ratio * C), H, W)
        x = self.bn2(self.conv2(ops.flatten(x, start_dim=0, end_dim=1))).view(T, B, C, H, W)
        x = x_feat + x
        return x


class MS_AllConvBlock(nn.Cell):  # standard conv
    def __init__(self, input_dim, mlp_ratio=4., sep_kernel_size=7, group=False):  # in_channels(out_channels), 内部扩张比例
        super().__init__()

        self.Conv = SepConv(dim=input_dim, kernel_size=sep_kernel_size)

        self.mlp_ratio = mlp_ratio
        self.conv1 = MS_StandardConv(input_dim, int(input_dim * mlp_ratio), 3)
        self.conv2 = MS_StandardConv(int(input_dim * mlp_ratio), input_dim, 3)

    # @get_local('x_feat')
    def construct(self, x):
        T, B, C, H, W = x.shape
        print("+++++++X1++++++++++++++++++++++++++", x)


        x = self.Conv(x) + x  # sepconv  pw+dw+pw

        x_feat = x

        x = self.conv1(x)
        x = self.conv2(x)
        x = x_feat + x

        return x


class MS_StandardConv(nn.Cell):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1):  # in_channels(out_channels), 内部扩张比例
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.s = s
        self.conv = nn.Conv2d(c1, c2, k, s,pad_mode ="pad",padding = autopad(k, p, d), group=g, has_bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.lif = mem_update()

    def construct(self, x):
        T, B, C, H, W = x.shape
        x = self.bn(self.conv(ops.flatten(self.lif(x), start_dim=0, end_dim=1))).reshape(T, B, self.c2, int(H / self.s), int(W / self.s))
        return x


class MS_DownSampling(nn.Cell):
    def __init__(self, in_channels=2, embed_dims=256, kernel_size=3, stride=2, padding=1, first_layer=True):
        super().__init__()
        self.encode_conv = nn.Conv2d(in_channels, embed_dims, kernel_size=kernel_size, stride=stride, padding=padding, pad_mode="pad")

        self.encode_bn = nn.BatchNorm2d(embed_dims)
        if not first_layer:
            self.encode_lif = mem_update()
        # self.pool = nn.MaxPool2d(kernel_size=2)

    def construct(self, x):
        T, B, _, _, _ = x.shape

        if hasattr(self, "encode_lif"):  # 如果不是第一层
            # x_pool = self.pool(x)
            x = self.encode_lif(x)


        x = self.encode_conv(ops.flatten(x, start_dim=0, end_dim=1))  #11111111111改到这里了，报错


        _, C, H, W = x.shape
        x = self.encode_bn(x).reshape(T, B, -1, H, W).contiguous()

        return x


class MS_GetT(nn.Cell):
    def __init__(self, in_channels=1, out_channels=1, T=1):
        super().__init__()
        self.T = T
        self.in_channels = in_channels

    def construct(self, x):

        x = (x.unsqueeze(0)).repeat(self.T,axis=0)
        print("+++++++X00++++++++++++++++++++++++++", x)
        return x


class MS_CancelT(nn.Cell):
    def __init__(self, in_channels=1, out_channels=1, T=2):
        super().__init__()
        self.T = T

    def construct(self, x):
        x = x.mean(0)
        return x


class SpikeConv(nn.Cell):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, group, dilation, activation)

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, pad_mode ="pad",padding = autopad(k, p, d), group=g, dilation=d, has_bias=False)
        self.lif = mem_update()
        self.bn = nn.BatchNorm2d(c2)
        self.s = s

    def construct(self, x):
        T, B, C, H, W = x.shape
        H_new = int(H / self.s)
        W_new = int(W / self.s)
        x = self.lif(x)

        x = self.bn(self.conv(ops.flatten(x, start_dim=0, end_dim=1))).reshape(T, B, -1, H_new, W_new)
        return x


class SpikeConvWithoutBN(nn.Cell):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, group, dilation, activation)

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, pad_mode ="pad",padding = autopad(k, p, d), group=g, dilation=d, has_bias=False)
        self.lif = mem_update()

        self.s = s
        # self.act = self.default_act if act is True else act if isinstance(act, nn.Cell) else nn.Identity()

    def construct(self, x):
        T, B, C, H, W = x.shape
        H_new = int(H / self.s)
        W_new = int(W / self.s)
        x = self.lif(x)
        x = self.conv(ops.flatten(x, start_dim=0, end_dim=1)).reshape(T, B, -1, H_new, W_new)
        return x


class SpikeSPPF(nn.Cell):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        p = int(k/2-0.5)
        self.cv1 = SpikeConv(c1, c_, 1, 1)
        self.cv2 = SpikeConv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, pad_mode='pad', padding=p)

    def construct(self, x):
        x = self.cv1(x)

        T, B, C, H, W = x.shape

        y1 = self.m(ops.flatten(x, start_dim=0, end_dim=1)).reshape(T, B, -1, H, W)
        y2 = self.m(ops.flatten(y1, start_dim=0, end_dim=1)).reshape(T, B, -1, H, W)
        y3 = self.m(ops.flatten(y2, start_dim=0, end_dim=1)).reshape(T, B, -1, H, W)
        return self.cv2(ops.cat((x, y1, y2, y3), 2))


class MS_Concat(nn.Cell):
    # Concatenate a list of tensors along dimension好的
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def construct(self, x):  # 这里输入x是一个list
        for i in range(len(x)):
            if x[i].dim() == 5:
                x[i] = x[i].mean(0)
        return ops.cat(x, self.d)


class DFL(nn.Cell):
    # Integral module of Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, has_bias=False)
        self.conv.weight.requires_grad = False
        self.c1 = c1
        self.softmax = ops.Softmax(axis=1)

    def construct(self, x):
        b, c, a = x.shape  # batch, channels, anchors
        x = self.softmax(x.view(b, 4, self.c1, a).swapaxes(2, 1))
        x = self.conv(x)
        x = x.view(b, 4, a)
        return x

    def initialize_conv_weight(self):
        self.conv.weight = ops.assign(
            self.conv.weight, Tensor(np.arange(self.c1).reshape((1, self.c1, 1, 1)), dtype=ms.float32)
        )

# class SpikeDFL(nn.Cell):
#     """
#     Integral module of Distribution Focal Loss (DFL).
#
#     Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
#     """
#
#     def __init__(self, c1=16):
#         """Initialize a convolutional layer with a given number of input channels."""
#         super().__init__()
#         self.conv = nn.Conv2d(c1, 1, 1).requires_grad_(False)
#         x = ops.arange(c1, ms.float)  #[0,1,2,...,15]
#         self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1)) #这里不是脉冲驱动的，但是是整数乘法
#         self.c1 = c1  #本质上就是个加权和。输入是每个格子的概率(小数)，权重是每个格子的位置(整数)
#         self.lif = mem_update()
#
#
#     def forward(self, x):
#         """Applies a transformer layer on input tensor 'x' and returns a tensor."""
#         b, c, a = x.shape  # batch, channels, anchors
#         # x = self.lif(x.view(1,b, 4, self.c1, a).transpose(3, 2))  #[T,B,C,A]
#         # x = x.mean(0) #[B,C,A]
#         # print("=================================")
#         # x = self.conv(x).view(b, 4, a) # #
#         # return x
#         # self.conv(ops.flatten(x, start_dim=0, end_dim=1))).reshape(T, b, -1, H_new, W_new)
#         # return self.conv(x).view(b, 4, a)
# #         print("weight:",self.conv.weight.data)
#         # return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1)).view(b, 4, a)  #原版
#         return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)  # 原版

class SpikeDetect(nn.Cell):
    """YOLOv8 Detect head for detection models."""
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None

    def __init__(self, nc=80, ch=()):
        """Initializes the YOLOv8 detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = mint.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels

        self.cv2 = nn.SequentialCell()
        self.cv3 = nn.SequentialCell()

        for x in ch:
            cv2_onelevel = nn.SequentialCell()
            cv2_onelevel.append(SpikeConv(x, c2, 3))
            cv2_onelevel.append(SpikeConv(c2, c2, 3))
            cv2_onelevel.append(SpikeConvWithoutBN(c2, 4 * self.reg_max, 1))
            self.cv2.append(cv2_onelevel)

        self.cv3 = nn.SequentialCell()
        for x in ch:
            cv3_onelevel = nn.SequentialCell()
            cv3_onelevel.append(SpikeConv(x, c3, 3))
            cv3_onelevel.append(SpikeConv(c3, c3, 3))
            cv3_onelevel.append(SpikeConvWithoutBN(c3, self.nc, 1))
            self.cv3.append(cv3_onelevel)
        # self.cv2 = nn.SequentialCell(
        #     nn.SequentialCell(SpikeConv(x, c2, 3), SpikeConv(c2, c2, 3), SpikeConvWithoutBN(c2, 4 * self.reg_max, 1)) for x in ch)
        # self.cv3 = nn.SequentialCell(nn.Sequential(SpikeConv(x, c3, 3), SpikeConv(c3, c3, 3), SpikeConvWithoutBN(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def construct(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        x = list(x)
        shape = x[0].mean(0).shape  # BCHW 这里x[0]是第一个检测头，和维度无关 推理：[1，2，64，32，84]  这里必须mean0，否则推理时用到shape会导致报错
        for i in range(self.nl):
            x[i] = ops.concat((self.cv2[i](x[i]), self.cv3[i](x[i])), 2)
            x[i] = x[i].mean(0)  #[2，144，32，684]  #这个地方有时候全是1.之后debug看看
        if self.training:
            x = tuple(x)
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in self.make_anchors(x, self.stride, 0.5))
            self.shape = shape

        x_cat = ops.concat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # avoid TF FlexSplitV ops
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1) #box: [B,reg_max * 4,anchors]
        dbox = self.dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides



        y = ops.concat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].conv.bias.data[:] = 1.0  # box
            b[-1].conv.bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)


    @staticmethod
    def make_anchors(feats, strides, grid_cell_offset=0.5):
        """Generate anchors from features."""
        anchor_points, stride_tensor = (), ()
        dtype = feats[0].dtype
        for i, stride in enumerate(strides):
            _, _, h, w = feats[i].shape
            sx = mnp.arange(w, dtype=dtype) + grid_cell_offset  # shift x
            sy = mnp.arange(h, dtype=dtype) + grid_cell_offset  # shift y
            # FIXME: Not supported on a specific model of machine
            sy, sx = meshgrid((sy, sx), indexing="ij")
            anchor_points += (ops.stack((sx, sy), -1).view(-1, 2),)
            stride_tensor += (ops.ones((h * w, 1), dtype) * stride,)
        return ops.concat(anchor_points), ops.concat(stride_tensor)

    @staticmethod
    def dist2bbox(distance, anchor_points, xywh=True, axis=-1):
        """Transform distance(ltrb) to box(xywh or xyxy)."""
        lt, rb = ops.split(distance, split_size_or_sections=2, axis=axis)
        x1y1 = anchor_points - lt
        x2y2 = anchor_points + rb
        if xywh:
            c_xy = (x1y1 + x2y2) / 2
            wh = x2y2 - x1y1
            return ops.concat((c_xy, wh), axis)  # xywh bbox
        return ops.concat((x1y1, x2y2), axis)  # xyxy bbox