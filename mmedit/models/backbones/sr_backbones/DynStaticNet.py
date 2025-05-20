import decord
from click import Parameter



decord.bridge.set_bridge('torch')
import torch.utils.data
import torch.nn.functional as F
from einops import rearrange
import torch
import torch.nn as nn
import warnings
import math
from functools import reduce
from operator import mul
import random
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops.layers.torch import Rearrange

from pytorch_wavelets import DWTForward



# 深度卷积 (Depthwise Convolution)
class DepthwiseConv2d(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(DepthwiseConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=bias)

    def forward(self, x):
        return self.depthwise(x)

# 逐点卷积 (Pointwise Convolution)
class PointwiseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super(PointwiseConv2d, self).__init__()
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        return self.pointwise(x)


class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=1.0):
        super(Conv2d_cd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight = Rearrange('c_in c_out k1 k2 -> c_in c_out (k1 k2)')(conv_weight)
        conv_weight_cd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3).fill_(0)
        conv_weight_cd[:, :, :] = conv_weight[:, :, :]
        conv_weight_cd[:, :, 4] = conv_weight[:, :, 4] - conv_weight[:, :, :].sum(2)
        conv_weight_cd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[3])(
            conv_weight_cd)
        return conv_weight_cd, self.conv.bias


class Conv2d_hd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=1.0):
        super(Conv2d_hd, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight_hd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3).fill_(0)
        conv_weight_hd[:, :, [0, 3, 6]] = conv_weight[:, :, :]
        conv_weight_hd[:, :, [2, 5, 8]] = -conv_weight[:, :, :]
        conv_weight_hd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[2])(
            conv_weight_hd)
        return conv_weight_hd, self.conv.bias


class Conv2d_vd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):
        super(Conv2d_vd, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight_vd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3).fill_(0)
        conv_weight_vd[:, :, [0, 1, 2]] = conv_weight[:, :, :]
        conv_weight_vd[:, :, [6, 7, 8]] = -conv_weight[:, :, :]
        conv_weight_vd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[2])(
            conv_weight_vd)
        return conv_weight_vd, self.conv.bias

def ActivationLayer(act_type):
    if act_type == 'gelu':
        act_layer = nn.GELU()

    return act_layer


def NormLayer(ch_width):

    norm_layer = nn.BatchNorm2d(num_features=ch_width)

    return norm_layer

class DSEConv(nn.Module):
    def __init__(self, dim):
        super(DSEConv, self).__init__()
        self.dim = dim
        self.conv1_1 = Conv2d_cd(dim, dim, 3, bias=True, groups=dim)
        self.conv1_2 = Conv2d_hd(dim, dim, 3, bias=True, groups=dim)
        self.conv1_3 = Conv2d_vd(dim, dim, 3, bias=True, groups=dim)
        self.conv1_4 = nn.Conv2d(dim, dim, 3, padding=1, bias=True, groups=dim)
        self.pointwise = PointwiseConv2d(dim, dim, bias=True)  # 逐点卷积
        self.convout = nn.Conv2d(dim, dim, 3, padding=1, bias=True)
        self.act = nn.GELU()

    def forward(self, x):
        x = x.permute(0,3,1,2)
        # print(x.shape)
        w1, b1 = self.conv1_1.get_weight()
        w2, b2 = self.conv1_2.get_weight()
        w3, b3 = self.conv1_3.get_weight()
        w4, b4 = self.conv1_4.weight, self.conv1_4.bias
        w = w1 + w2 + w3 + w4
        b = b1 + b2 + b3 + b4

        res = nn.functional.conv2d(input=x, weight=w, bias=b, stride=1, padding=1, groups=self.dim)
        res = self.pointwise(res)
        res = self.act(res)
        res = res + x
        res = self.convout(res)
        res = res + x
        return res.permute(0,2,3,1)




class Spacial3D(nn.Module):

    def __init__(self, dim=48, window_size=(1,8,8), num_heads=8, dec=False, qkv_bias=False, qk_scale=None, attn_drop=0.01, proj_drop=0.01,):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wd, Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.dec = dec
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.query = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):

        B, T, W, N, C = x.shape
        query_leran = nn.Parameter(torch.randn(B, T, W, 128, C)).cuda()
        query_leran = self.query(query_leran).reshape(B*T, 128,  self.num_heads, W * C // self.num_heads).permute(0, 2, 1, 3)

        qkv = self.qkv(x).reshape(B*T, N, 3, self.num_heads, W * C // self.num_heads).permute(2, 0, 3, 1, 4)
                                                                                          # 3, B_, nH, W, N*C
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, W, nH, N*C

        if self.dec==True:
            q = torch.nn.functional.interpolate(query_leran, (q.shape[2], q.shape[3]))

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B*T // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B*T, W, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Temporal3D(nn.Module):

    def __init__(self, dim=48, window_size=(1,8,8), num_heads=8, dec=False, qkv_bias=False, qk_scale=None, attn_drop=0.01, proj_drop=0.01):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wd, Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.query = nn.Linear(dim, dim, bias=qkv_bias)
        self.dec = dec
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):

        B, T, W, N, C = x.shape
        query_leran = nn.Parameter(torch.randn(B, T, W, 128, C)).cuda()
        query_leran = self.query(query_leran).reshape(B, T, 128,  self.num_heads, W * C // self.num_heads).permute(0, 3, 2, 1, 4)
        qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, W * C // self.num_heads).permute(3, 0, 4, 2, 1, 5)
                                                                                          # 3, B, nH, W, T, N*C
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, W, nH, N, C
        # print(q.shape,query_leran.shape)
        if self.dec==True:
            q = torch.nn.functional.interpolate(query_leran, (q.shape[2], q.shape[3], q.shape[4]))

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B*T // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B*T, W, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class DynStaticformerBlock3D(nn.Module):

    def __init__(self, dim=32, num_heads=8, window_size=(1, 8, 8), dec=False, mlp_ratio=4., shift_size=(0, 0, 0),
                 qkv_bias=True, qk_scale=None, drop=0.01, attn_drop=0.01, norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim//2)
        self.spatial_attn = Spacial3D(
            dim//2, window_size=self.window_size, num_heads=num_heads,dec=dec,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.temporal_attn = Temporal3D(
            dim//2, window_size=self.window_size, num_heads=num_heads,dec=dec,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.DSEConv = DSEConv(dim)
        self.fusion_drop = nn.Dropout(0.01)
        self.norm2 = norm_layer(dim//2)

        self.Mlp = DGFN(dim//2)

    def Temporal_Attn(self, x):

        B, D, H, W, C = x.shape
        PAD1 = H
        PAD2 = W
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)
        # print("window_size:",window_size,"shift_size:",shift_size)
        x = self.norm1(x)
        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        x_skip = x
        # print("after pad:",x.shape)
        _, Dp, Hp, Wp, _ = x.shape

        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = None
        else:
            shifted_x = x
            attn_mask = None
        # partition windows
        x_windows = window_partition(shifted_x, window_size)  # B*T, nW, Wd*Wh*Ww, C
        T, nW, W, C= x_windows.size()
        x_windows = x_windows.view(-1, D, nW, W, C)
        # print('x_windows',x_windows.shape)

        # W-MSA/SW-MSA
        attn_windows = self.temporal_attn(x_windows, mask=attn_mask)  # B*T, nW, Wd*Wh*Ww, C
        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size + (C,)))
        x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)  # B D' H' W' C
        x = x + x_skip
        x = x[:, :, :PAD1, :PAD2, :]

        return x

    def Spatial_Attn(self, x):

        B, D, H, W, C = x.shape
        PAD1 = H
        PAD2 = W
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)
        # print("window_size:",window_size,"shift_size:",shift_size)
        x = self.norm1(x)
        tensor_bottom = x
        # print('tensor_bottom:', tensor_bottom.shape)
        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        x_skip = x
        # print("after pad:",x.shape)
        _, Dp, Hp, Wp, _ = x.shape

        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = None
        else:
            shifted_x = x
            attn_mask = None
        # partition windows
        x_windows = window_partition(shifted_x, window_size)  # B*T, nW, Wd*Wh*Ww, C
        T, nW, W, C= x_windows.size()
        x_windows = x_windows.view(-1, D, nW, W, C)
        # print('x_windows',x_windows.shape)

        # W-MSA/SW-MSA
        attn_windows = self.spatial_attn(x_windows, mask=attn_mask)  # B*T, nW, Wd*Wh*Ww, C
        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size + (C,)))
        x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)  # B D' H' W' C
        x = x + x_skip
        x = x[:, :, :PAD1, :PAD2, :]

        return x

    def DGFNMlp(self, x):
        return self.Mlp(self.norm2(x))

    def forward(self, x):
        x = rearrange(x, 'b t c h w -> b t h w c')
        b, f, h, w, c = x.shape
        output_frames = []
        for i in range(f):
            frame = x[:, i, :, :, :]  # 提取第 i 帧, shape: (b, h, w, c)
            processed_frame = self.DSEConv(frame)  # 通过网络处理
            output_frames.append(processed_frame)
        x_img = torch.stack(output_frames, dim=1)  # 输出形状为 (b, f, c, h, w)
        x1, x2 = torch.split(x_img, c // 2, dim=4)

        x1 = x1 + self.Spatial_Attn(x1)
        x1 = x1 + self.DGFNMlp(x1)
        x2 = x2 + self.Temporal_Attn(x2)
        x2 = x2 + self.DGFNMlp(x2)
        out = torch.cat((x1, x2), dim=4)
        out = rearrange(out, 'b t h w c -> b t c h w')
        return out


class DGFN(nn.Module):
    def __init__(self, dim):
        super(DGFN, self).__init__()

        self.project_in = nn.Conv2d(dim, dim * 2, kernel_size=1)
        # 分组卷积
        self.dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1,
                                groups=dim * 2)
        #self.project_out = nn.Conv2d(dim, dim, kernel_size=1)
    def forward(self, x):
        b, f, h, w, c = x.size()
        x = x.view(-1, h, w, c).permute(0, 3, 1, 2).contiguous()
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)  # 在通道维度分为两部分
        x = F.gelu(x1) * x2   # 门控机制
        #x = self.project_out(x)
        _, c, h, w = x.shape
        x = x.view(b, f, c, h, w).permute(0, 1, 3, 4, 2).contiguous()

        return x



def window_partition(x, window_size):
    """ Partition the input into windows. Attention will be conducted within the windows.
    Args:
        x: (B, D, H, W, C)
        window_size (tuple[int]): window size
    Returns:
        windows: (B*num_windows, window_size*window_size, C)
    """
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2],
               window_size[2], C)
    # print(x.shape)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(B*D, -1, reduce(mul, window_size), C)

    return windows


def window_reverse(windows, window_size, B, D, H, W):
    """ Reverse windows back to the original input. Attention was conducted within the windows.
    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, D, H, W, C)
    """
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1],
                     window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)

    return x

def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)

def _no_grad_trunc_normal_(tensor, mean, std, a, b):

    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            'mean is more than 2 std from [a, b] in nn.init.trunc_normal_. '
            'The distribution of values may be incorrect.',
            stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        low = norm_cdf((a - mean) / std)
        up = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [low, up], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * low - 1, 2 * up - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor

class DynNet(nn.Module):

    def __init__(self, dim=48, num_heads=8, window_size=(1, 8, 8), depth=2, dec=False,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0.01, attn_drop=0.01,
                  norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)

        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            DynStaticformerBlock3D(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else self.shift_size,
                dec=dec,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
            )
            for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x



def ActivationLayer(act_type):
    if act_type == 'gelu':
        act_layer = nn.GELU()

    return act_layer


def NormLayer(ch_width):

    norm_layer = nn.BatchNorm2d(num_features=ch_width)

    return norm_layer


class Embeddings(nn.Module):
    def __init__(self, dim1, dim2):
        super(Embeddings, self).__init__()

        self.activation = nn.LeakyReLU(0.2, True)
        self.layer = nn.Sequential(
            nn.Conv2d(dim1, dim2, kernel_size=3, padding=1),
            self.activation,
        )

    def forward(self, x):
        b, f, c, h, w = x.size()
        x = x.view(-1, c, h, w)
        hx = self.layer(x)
        _, c, h, w = hx.shape
        hx = hx.view(b, f, c, h, w)

        return hx

class UpSample(nn.Module):
    def __init__(self, dim, strds=2, r=2):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim//r*strds**2, 3, 1, 1, bias=False)
        self.up_scale = nn.PixelShuffle(strds)
        self.norm = NormLayer(dim//r)
        self.act = ActivationLayer('gelu')

    def forward(self, x):
        b, f, c, h, w = x.size()
        x = x.view(-1, c, h, w)

        x = self.conv(x)
        x = self.up_scale(x)
        x = self.act(self.norm((x)))

        _, c, h, w = x.shape
        x = x.view(b, f, c, h, w)
        return x



class Down_wt(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down_wt, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
                                    nn.Conv2d(in_ch*4, out_ch, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(out_ch),
                                    nn.ReLU(inplace=True),
                                    )
    def forward(self, x):
        b, f, c, h, w = x.size()
        x = x.view(-1, c, h, w)

        yL, yH = self.wt(x)
        y_HL = yH[0][:,:,0,::]
        y_LH = yH[0][:,:,1,::]
        y_HH = yH[0][:,:,2,::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv_bn_relu(x)
        _, c, h, w = x.shape
        x = x.view(b, f, c, h, w)
        return x


class Reduce_chans(nn.Module):
    def __init__(self, dim1, dim2):
        super(Reduce_chans, self).__init__()

        self.Reduce_chans = nn.Conv2d(dim1, dim2, kernel_size=1)

    def forward(self, x):
        b, f, c, h, w = x.size()
        x = x.view(-1, c, h, w)
        hx = self.Reduce_chans(x)
        _, c, h, w = hx.shape
        hx = hx.view(b, f, c, h, w)

        return hx

from mmedit.models.registry import BACKBONES
@BACKBONES.register_module()
class DynStaticNet(nn.Module):

    def __init__(self, in_chans=3, dim=24, num_heads=[1, 2, 4], depth=[4, 6, 8], dec=False
                ):
        super().__init__()
        # input
        self.conv_input = Embeddings(in_chans,dim)

        # encoder

        self.vid_e1 = DynNet(dim=dim, num_heads=num_heads[0], depth=depth[0])

        self.Down1 = Down_wt(dim, dim * 2 ** 1)

        self.vid_e2 = DynNet(dim=dim * 2 ** 1, num_heads=num_heads[1], depth=depth[1])

        self.Down2 = Down_wt(dim * 2 ** 1, dim * 2 ** 2)

        # middle
        self.vid_m3 = DynNet(dim=dim * 2 ** 2, num_heads=num_heads[2], depth=depth[2])


        # decoder
        self.Up2 = UpSample(dim=dim * 2 ** 2)
        self.reduce_chan2 = Reduce_chans(dim * 2 ** 2, dim * 2 ** 1)
        self.vid_d2 = DynNet(dim=dim * 2 ** 1, num_heads=num_heads[1], depth=depth[1],dec=True)


        self.Up1 = UpSample(dim=dim * 2 ** 1)
        self.reduce_chan1 = Reduce_chans(dim * 2 ** 1, dim * 2 ** 0)
        self.vid_d1 = DynNet(dim=dim, num_heads=num_heads[0], depth=depth[0],dec=True)

        # refinement

        self.refinement = DynNet(dim=dim, num_heads=2, depth=4,dec=True)

        # output
        self.conv_out = Embeddings(dim, in_chans)



    def forward(self, x):
        # input
        input = x

        # encoder
        # layer1
        x = self.conv_input(x)

        V_e1 = self.vid_e1(x)
        V1_skip = V_e1



        # layer2
        V_e2 = self.Down1(V_e1)
        V_e2= self.vid_e2(V_e2)
        V2_skip = V_e2



        # layer3
        V_m3 = self.Down2(V_e2)
        V_m3 = self.vid_m3(V_m3)



        #decoder
        # layer2
        V_d2 = self.Up2(V_m3)
        V_d2 = torch.cat([V_d2, V2_skip], 2)
        V_d2 = self.reduce_chan2(V_d2)
        V_d2 = self.vid_d2(V_d2)


        # layer1
        V_d1 = self.Up1(V_d2)
        V_d1 = torch.cat([V_d1, V1_skip], 2)
        V_d1 = self.reduce_chan1(V_d1)
        V_d1 = self.vid_d1(V_d1)


        refine = self.refinement(V_d1)


        # output
        x = self.conv_out(refine)
        return x + input

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.
        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Default: None.
            strict (bool, optional): Whether strictly load the pretrained
                model. Default: True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')



