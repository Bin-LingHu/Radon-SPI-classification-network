"""
This RadonSPIFly Classification codes are based on Mambaout framework.
Some implementations are modified from:
timm (https://github.com/rwightman/pytorch-image-models),
MambaOut (https://github.com/yuweihao/MambaOut),
MetaFormer (https://github.com/sail-sg/metaformer),
InceptionNeXt (https://github.com/sail-sg/inceptionnext)
"""
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': 1.0, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'mambaout_femto': _cfg(
        url='https://github.com/yuweihao/MambaOut/releases/download/model/mambaout_femto.pth'),
    'mambaout_kobe': _cfg(
        url='https://github.com/yuweihao/MambaOut/releases/download/model/mambaout_kobe.pth'),
    'mambaout_tiny': _cfg(
        url='https://github.com/yuweihao/MambaOut/releases/download/model/mambaout_tiny.pth'),
    'mambaout_small': _cfg(
        url='https://github.com/yuweihao/MambaOut/releases/download/model/mambaout_small.pth'),
    'mambaout_base': _cfg(
        url='https://github.com/yuweihao/MambaOut/releases/download/model/mambaout_base.pth'),
}


class StemLayer(nn.Module):
    r""" Code modified from InternImage:
        https://github.com/OpenGVLab/InternImage
    """

    def __init__(self,
                 in_channels=3,
                 out_channels=96,
                 act_layer=nn.GELU,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,
                               out_channels // 2,
                               kernel_size=3,
                               stride=2,
                               padding=1)
        self.norm1 = norm_layer(out_channels // 2)
        self.act = act_layer()
        self.conv2 = nn.Conv2d(out_channels // 2,
                               out_channels,
                               kernel_size=3,
                               stride=2,
                               padding=1)
        self.norm2 = norm_layer(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1) # NCHW-> NHWC 归一化层输入形状
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.act(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm2(x)
        return x


class DownsampleLayer(nn.Module):
    r""" Code modified from InternImage:
        https://github.com/OpenGVLab/InternImage
    """
    def __init__(self, in_channels=96, out_channels=198, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=3,
                              stride=2,
                              padding=1)
        self.norm = norm_layer(out_channels)

    def forward(self, x):
        x = self.conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = self.norm(x)
        return x


class MlpHead(nn.Module):
    """ MLP classification head
    """
    def __init__(self, dim, num_classes=1000, act_layer=nn.GELU, mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), head_dropout=0., bias=True):
        super().__init__()
        hidden_features = int(mlp_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden_features, bias=bias)
        self.act = act_layer()
        self.norm = norm_layer(hidden_features)
        self.fc2 = nn.Linear(hidden_features, num_classes, bias=bias)
        self.head_dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.head_dropout(x)
        x = self.fc2(x)
        return x


class GatedCNNBlock(nn.Module):
    r""" Our implementation of Gated CNN Block: https://arxiv.org/pdf/1612.08083
    Args: 
        conv_ratio: control the number of channels to conduct depthwise convolution.
            Conduct convolution on partial channels can improve practical efficiency.
            The idea of partial channels is from ShuffleNet V2 (https://arxiv.org/abs/1807.11164) and 
            also used by InceptionNeXt (https://arxiv.org/abs/2303.16900) and FasterNet (https://arxiv.org/abs/2303.03667)
    """
    def __init__(self, dim, expansion_ratio=8/3, kernel_size=7, conv_ratio=1.0,
                 norm_layer=partial(nn.LayerNorm,eps=1e-6), 
                 act_layer=nn.GELU,
                 drop_path=0.,
                 **kwargs):
        super().__init__()
        self.norm = norm_layer(dim)
        hidden = int(expansion_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden * 2)
        self.act = act_layer()
        conv_channels = int(conv_ratio * dim)
        self.split_indices = (hidden, hidden - conv_channels, conv_channels) #输出分为三部分：门控信号(g)、直接传递特征(i)和卷积特征(c)
        self.conv = nn.Conv2d(conv_channels, conv_channels, kernel_size=kernel_size, padding=kernel_size//2, groups=conv_channels)
        self.fc2 = nn.Linear(hidden, dim) #将处理后的特征压缩回原始维度
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x # [B, H, W, C]
        x = self.norm(x)
        g, i, c = torch.split(self.fc1(x), self.split_indices, dim=-1) #通过fc1扩展并分割特征为三部分(g, i, c)
        c = c.permute(0, 3, 1, 2) # [B, H, W, C] -> [B, C, H, W]
        c = self.conv(c)
        c = c.permute(0, 2, 3, 1) # [B, C, H, W] -> [B, H, W, C]
        x = self.fc2(self.act(g) * torch.cat((i, c), dim=-1))  #将g作为门控信号，与i和c的拼接结果相乘
        x = self.drop_path(x)
        return x + shortcut






r"""
downsampling (stem) for the first stage is two layer of conv with k3, s2 and p1
downsamplings for the last 3 stages is a layer of conv with k3, s2 and p1
DOWNSAMPLE_LAYERS_FOUR_STAGES format: [Downsampling, Downsampling, Downsampling, Downsampling]
use `partial` to specify some arguments
"""
DOWNSAMPLE_LAYERS_FOUR_STAGES = [StemLayer] + [DownsampleLayer]*3


class MambaOut(nn.Module):
    r""" MetaFormer
        A PyTorch impl of : `MetaFormer Baselines for Vision`  -
          https://arxiv.org/abs/2210.13452

    Args:
        in_chans (int): Number of input image channels. Default: 3.
        num_classes (int): Number of classes for classification head. Default: 1000.
        depths (list or tuple): Number of blocks at each stage. Default: [3, 3, 9, 3].
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 576].
        downsample_layers: (list or tuple): Downsampling layers before each stage.
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        output_norm: norm before classifier head. Default: partial(nn.LayerNorm, eps=1e-6).
        head_fn: classification head. Default: nn.Linear.
        head_dropout (float): dropout for MLP classifier. Default: 0.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3],
                 dims=[96, 192, 384, 576],
                 downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 act_layer=nn.GELU,
                 conv_ratio=1.0,
                 kernel_size=7,
                 drop_path_rate=0.,
                 output_norm=partial(nn.LayerNorm, eps=1e-6), 
                 head_fn=MlpHead,
                 head_dropout=0.0, 
                 **kwargs,
                 ):
        super().__init__()
        self.num_classes = num_classes

        if not isinstance(depths, (list, tuple)):  
            depths = [depths] # it means the model has only one stage  如果只提供一个值，则将其转换为列表，表示只有一个阶段。
        if not isinstance(dims, (list, tuple)):
            dims = [dims]

        num_stage = len(depths)
        self.num_stage = num_stage

        if not isinstance(downsample_layers, (list, tuple)):
            downsample_layers = [downsample_layers] * num_stage
        down_dims = [in_chans] + dims
        self.downsample_layers = nn.ModuleList(
            [downsample_layers[i](down_dims[i], down_dims[i+1]) for i in range(num_stage)]
        )

        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.stages = nn.ModuleList()
        cur = 0
        for i in range(num_stage):
            stage = nn.Sequential(
                *[GatedCNNBlock(dim=dims[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                kernel_size=kernel_size,
                conv_ratio=conv_ratio,
                drop_path=dp_rates[cur + j],
                ) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = output_norm(dims[-1])

        if head_dropout > 0.0:
            self.head = head_fn(dims[-1], num_classes, head_dropout=head_dropout)
        else:
            self.head = head_fn(dims[-1], num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'norm'}

    def forward_features(self, x):
        for i in range(self.num_stage):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([1, 2])) # (B, H, W, C) -> (B, C)  #x.mean([1, 2])：对特征图进行空间维度的平均池化

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


#------------------------
# DeepLBP and DeepLine
#------------------------
class Get_ltpe(nn.Module):
    def __init__(self):
        super(Get_ltpe, self).__init__()
        kernel7 = [[-1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0]]
        kernel6 = [[0, -1, 0],
                   [0, 1, 0],
                   [0, 0, 0]]
        kernel5 = [[0, 0, -1],
                  [0, 1, 0],
                  [0, 0, 0]]
        kernel4 = [[0, 0, 0],
                  [0, 1, -1],
                  [0, 0, 0]]
        kernel3 = [[0, 0, 0],
                  [0, 1, 0],
                  [0, 0, -1]]
        kernel2 = [[0, 0, 0],
                  [0, 1, 0],
                  [0, -1, 0]]
        kernel1 = [[0, 0, 0],
                  [0, 1, 0],
                  [-1, 0, 0]]
        kernel0 = [[0, 0, 0],
                  [-1, 1, 0],
                  [0, 0, 0]]

        kernel7 = torch.cuda.FloatTensor(kernel7).unsqueeze(0).unsqueeze(0)
        kernel6 = torch.cuda.FloatTensor(kernel6).unsqueeze(0).unsqueeze(0)
        kernel5 = torch.cuda.FloatTensor(kernel5).unsqueeze(0).unsqueeze(0)
        kernel4 = torch.cuda.FloatTensor(kernel4).unsqueeze(0).unsqueeze(0)
        kernel3 = torch.cuda.FloatTensor(kernel3).unsqueeze(0).unsqueeze(0)
        kernel2 = torch.cuda.FloatTensor(kernel2).unsqueeze(0).unsqueeze(0)
        kernel1 = torch.cuda.FloatTensor(kernel1).unsqueeze(0).unsqueeze(0)
        kernel0 = torch.cuda.FloatTensor(kernel0).unsqueeze(0).unsqueeze(0)

        self.weight_7 = nn.Parameter(data=kernel7, requires_grad=False)
        self.weight_6 = nn.Parameter(data=kernel6, requires_grad=False)
        self.weight_5 = nn.Parameter(data=kernel5, requires_grad=False)
        self.weight_4 = nn.Parameter(data=kernel4, requires_grad=False)
        self.weight_3 = nn.Parameter(data=kernel3, requires_grad=False)
        self.weight_2 = nn.Parameter(data=kernel2, requires_grad=False)
        self.weight_1 = nn.Parameter(data=kernel1, requires_grad=False)
        self.weight_0 = nn.Parameter(data=kernel0, requires_grad=False)

        weight_list = []
        weight_list.append(self.weight_0),weight_list.append(self.weight_1),weight_list.append(self.weight_2),weight_list.append(self.weight_3)
        weight_list.append(self.weight_4),weight_list.append(self.weight_5),weight_list.append(self.weight_6),weight_list.append(self.weight_7)
        self.weights = weight_list
        self.norm = torch.nn.InstanceNorm2d(1)

    def forward(self, x):

        x_gray = (0.3 * x[:, 0] + 0.59 * x[:, 1] + 0.11 * x[:, 2]).unsqueeze(1)
        out = torch.zeros_like(x_gray)

        for j in range(8):
            x_ltpe = F.conv2d(x_gray, self.weights[j], padding=1)
            x_ltpe = (x_ltpe +1)*0.5
            out = out + x_ltpe*(2**j)/255
        out = self.norm(out)
        out = torch.cat((out, out, out), dim=1)

        return out




class Block_ltpe(nn.Module):
    def __init__(self):
        super(Block_ltpe, self).__init__()
        kernel7 = [[-1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0]]
        kernel6 = [[0, -1, 0],
                   [0, 1, 0],
                   [0, 0, 0]]
        kernel5 = [[0, 0, -1],
                  [0, 1, 0],
                  [0, 0, 0]]
        kernel4 = [[0, 0, 0],
                  [0, 1, -1],
                  [0, 0, 0]]
        kernel3 = [[0, 0, 0],
                  [0, 1, 0],
                  [0, 0, -1]]
        kernel2 = [[0, 0, 0],
                  [0, 1, 0],
                  [0, -1, 0]]
        kernel1 = [[0, 0, 0],
                  [0, 1, 0],
                  [-1, 0, 0]]
        kernel0 = [[0, 0, 0],
                  [-1, 1, 0],
                  [0, 0, 0]]

        kernel7 = torch.cuda.FloatTensor(kernel7).unsqueeze(0).unsqueeze(0)
        kernel6 = torch.cuda.FloatTensor(kernel6).unsqueeze(0).unsqueeze(0)
        kernel5 = torch.cuda.FloatTensor(kernel5).unsqueeze(0).unsqueeze(0)
        kernel4 = torch.cuda.FloatTensor(kernel4).unsqueeze(0).unsqueeze(0)
        kernel3 = torch.cuda.FloatTensor(kernel3).unsqueeze(0).unsqueeze(0)
        kernel2 = torch.cuda.FloatTensor(kernel2).unsqueeze(0).unsqueeze(0)
        kernel1 = torch.cuda.FloatTensor(kernel1).unsqueeze(0).unsqueeze(0)
        kernel0 = torch.cuda.FloatTensor(kernel0).unsqueeze(0).unsqueeze(0)

        self.weight_7 = nn.Parameter(data=kernel7, requires_grad=False)
        self.weight_6 = nn.Parameter(data=kernel6, requires_grad=False)
        self.weight_5 = nn.Parameter(data=kernel5, requires_grad=False)
        self.weight_4 = nn.Parameter(data=kernel4, requires_grad=False)
        self.weight_3 = nn.Parameter(data=kernel3, requires_grad=False)
        self.weight_2 = nn.Parameter(data=kernel2, requires_grad=False)
        self.weight_1 = nn.Parameter(data=kernel1, requires_grad=False)
        self.weight_0 = nn.Parameter(data=kernel0, requires_grad=False)

        weight_list = []
        weight_list.append(self.weight_0),weight_list.append(self.weight_1),weight_list.append(self.weight_2),weight_list.append(self.weight_3)
        weight_list.append(self.weight_4),weight_list.append(self.weight_5),weight_list.append(self.weight_6),weight_list.append(self.weight_7)
        self.weights = weight_list
        self.norm = torch.nn.InstanceNorm2d(1)

    def forward(self, x):

        #x_gray = (0.3 * x[:, 0] + 0.59 * x[:, 1] + 0.11 * x[:, 2]).unsqueeze(1)
        out = torch.zeros_like(x)

        for j in range(8):
            x_ltpe = F.conv2d(x, self.weights[j], padding=1)
            x_ltpe = (x_ltpe +1)*0.5
            out = out + x_ltpe*(2**j)/255
        out = self.norm(out)
        #out = torch.cat((out, out, out), dim=1)

        return out



class Block_ltpe4feature(nn.Module):
    def __init__(self):
        super(Block_ltpe4feature, self).__init__()
        kernel7 = [[-1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0]]
        kernel6 = [[0, -1, 0],
                   [0, 1, 0],
                   [0, 0, 0]]
        kernel5 = [[0, 0, -1],
                  [0, 1, 0],
                  [0, 0, 0]]
        kernel4 = [[0, 0, 0],
                  [0, 1, -1],
                  [0, 0, 0]]
        kernel3 = [[0, 0, 0],
                  [0, 1, 0],
                  [0, 0, -1]]
        kernel2 = [[0, 0, 0],
                  [0, 1, 0],
                  [0, -1, 0]]
        kernel1 = [[0, 0, 0],
                  [0, 1, 0],
                  [-1, 0, 0]]
        kernel0 = [[0, 0, 0],
                  [-1, 1, 0],
                  [0, 0, 0]]

        kernel7 = torch.cuda.FloatTensor(kernel7).unsqueeze(0).unsqueeze(0)
        kernel6 = torch.cuda.FloatTensor(kernel6).unsqueeze(0).unsqueeze(0)
        kernel5 = torch.cuda.FloatTensor(kernel5).unsqueeze(0).unsqueeze(0)
        kernel4 = torch.cuda.FloatTensor(kernel4).unsqueeze(0).unsqueeze(0)
        kernel3 = torch.cuda.FloatTensor(kernel3).unsqueeze(0).unsqueeze(0)
        kernel2 = torch.cuda.FloatTensor(kernel2).unsqueeze(0).unsqueeze(0)
        kernel1 = torch.cuda.FloatTensor(kernel1).unsqueeze(0).unsqueeze(0)
        kernel0 = torch.cuda.FloatTensor(kernel0).unsqueeze(0).unsqueeze(0)

        self.weight_7 = nn.Parameter(data=kernel7, requires_grad=False)
        self.weight_6 = nn.Parameter(data=kernel6, requires_grad=False)
        self.weight_5 = nn.Parameter(data=kernel5, requires_grad=False)
        self.weight_4 = nn.Parameter(data=kernel4, requires_grad=False)
        self.weight_3 = nn.Parameter(data=kernel3, requires_grad=False)
        self.weight_2 = nn.Parameter(data=kernel2, requires_grad=False)
        self.weight_1 = nn.Parameter(data=kernel1, requires_grad=False)
        self.weight_0 = nn.Parameter(data=kernel0, requires_grad=False)

        weight_list = []
        weight_list.append(self.weight_0),weight_list.append(self.weight_1),weight_list.append(self.weight_2),weight_list.append(self.weight_3)
        weight_list.append(self.weight_4),weight_list.append(self.weight_5),weight_list.append(self.weight_6),weight_list.append(self.weight_7)
        self.weights = weight_list
        self.norm = torch.nn.InstanceNorm2d(1)

    def forward(self, x):
        B, C, H, W = x.shape
        #x_gray = (0.3 * x[:, 0] + 0.59 * x[:, 1] + 0.11 * x[:, 2]).unsqueeze(1)
        out = torch.zeros_like(x)

        for j in range(8):
            w_expanded = self.weights[j].repeat(C, 1, 1, 1)
            x_ltpe = F.conv2d(x, w_expanded, padding=1, groups=C)
            x_ltpe = (x_ltpe +1)*0.5
            out = out + x_ltpe*(2**j)/255
        out = self.norm(out)
        #out = torch.cat((out, out, out), dim=1)

        return out



class Get_line(nn.Module):
    def __init__(self):
        super(Get_line, self).__init__()
        kernel3 = [[-0.5, 0, 0],
                  [0, 1, 0],
                  [0, 0, -0.5]]
        kernel2 = [[0, -0.5, 0],
                  [0, 1, 0],
                  [0, -0.5, 0]]
        kernel1 = [[0, 0, -0.5],
                  [0, 1, 0],
                  [-0.5, 0, 0]]
        kernel0 = [[0, 0, 0],
                  [-0.5, 1, -0.5],
                  [0, 0, 0]]
   
        kernel3 = torch.cuda.FloatTensor(kernel3).unsqueeze(0).unsqueeze(0)
        kernel2 = torch.cuda.FloatTensor(kernel2).unsqueeze(0).unsqueeze(0)
        kernel1 = torch.cuda.FloatTensor(kernel1).unsqueeze(0).unsqueeze(0)
        kernel0 = torch.cuda.FloatTensor(kernel0).unsqueeze(0).unsqueeze(0)

        self.weight_3 = nn.Parameter(data=kernel3, requires_grad=False)
        self.weight_2 = nn.Parameter(data=kernel2, requires_grad=False)
        self.weight_1 = nn.Parameter(data=kernel1, requires_grad=False)
        self.weight_0 = nn.Parameter(data=kernel0, requires_grad=False)

        weight_list = []
        weight_list.append(self.weight_0),weight_list.append(self.weight_1),weight_list.append(self.weight_2),weight_list.append(self.weight_3)
        self.weights = weight_list
        self.norm = torch.nn.InstanceNorm2d(1)

    def forward(self, x):

        x_gray = (0.3 * x[:, 0] + 0.59 * x[:, 1] + 0.11 * x[:, 2]).unsqueeze(1)
        out = torch.zeros_like(x_gray)

        for j in range(4):
            x_ltpe = F.conv2d(x_gray, self.weights[j], padding=1)
            x_ltpe = (x_ltpe +1)*0.5
            out = out + x_ltpe*(2**j)/15
        out = self.norm(out)
        out = torch.cat((out, out, out), dim=1)

        return out



class Block_line4feature(nn.Module):
    def __init__(self):
        super(Block_line4feature, self).__init__()
        kernel3 = [[-0.5, 0, 0],
                  [0, 1, 0],
                  [0, 0, -0.5]]
        kernel2 = [[0, -0.5, 0],
                  [0, 1, 0],
                  [0, -0.5, 0]]
        kernel1 = [[0, 0, -0.5],
                  [0, 1, 0],
                  [-0.5, 0, 0]]
        kernel0 = [[0, 0, 0],
                  [-0.5, 1, -0.5],
                  [0, 0, 0]]
   
        kernel3 = torch.cuda.FloatTensor(kernel3).unsqueeze(0).unsqueeze(0)
        kernel2 = torch.cuda.FloatTensor(kernel2).unsqueeze(0).unsqueeze(0)
        kernel1 = torch.cuda.FloatTensor(kernel1).unsqueeze(0).unsqueeze(0)
        kernel0 = torch.cuda.FloatTensor(kernel0).unsqueeze(0).unsqueeze(0)

        self.weight_3 = nn.Parameter(data=kernel3, requires_grad=False)
        self.weight_2 = nn.Parameter(data=kernel2, requires_grad=False)
        self.weight_1 = nn.Parameter(data=kernel1, requires_grad=False)
        self.weight_0 = nn.Parameter(data=kernel0, requires_grad=False)

        weight_list = []
        weight_list.append(self.weight_0),weight_list.append(self.weight_1),weight_list.append(self.weight_2),weight_list.append(self.weight_3)
        self.weights = weight_list
        self.norm = torch.nn.InstanceNorm2d(1)

    def forward(self, x):
        B, C, H, W = x.shape
        #x_gray = (0.3 * x[:, 0] + 0.59 * x[:, 1] + 0.11 * x[:, 2]).unsqueeze(1)
        out = torch.zeros_like(x)

        for j in range(4):
            w_expanded = self.weights[j].repeat(C, 1, 1, 1)
            x_ltpe = F.conv2d(x, w_expanded, padding=1, groups=C)
            x_ltpe = (x_ltpe +1)*0.5
            out = out + x_ltpe*(2**j)/15
        out = self.norm(out)
        #out = torch.cat((out, out, out), dim=1)
        return out



class LightSA(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), "Kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        
        # 用单层卷积计算空间注意力（减少参数量）
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]
        concat = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W] 
        attn = self.conv(concat)  # [B, 1, H, W]
        return x * self.sigmoid(attn)  # 注意力加权


class LightweightLinearAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        
        self.key = nn.Linear(channels, channels // reduction, bias=False)
        self.query = nn.Linear(channels, channels // reduction, bias=False)
        self.value = nn.Linear(channels, channels, bias=False)
        
        self.gamma = nn.Parameter(torch.tensor(0.1))  # 可学习的缩放因子

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1).permute(0, 2, 1)  # [B, H*W, C]
        
        # 线性投影（降低维度）
        k = self.key(x_flat)  # [B, H*W, C//r]
        q = self.query(x_flat)  # [B, H*W, C//r]
        v = self.value(x_flat)  # [B, H*W, C]
        
        # 线性注意力计算
        attn = torch.softmax(q @ k.transpose(1, 2) / (C**0.5), dim=-1)  # [B, H*W, H*W]
        out = attn @ v  # [B, H*W, C]
        
        # 恢复形状并加权
        out = out.permute(0, 2, 1).view(B, C, H, W)
        return x + self.gamma * out


#--------------------------
# Refined MambaBlock with DeepLine/DeepLBP module
#--------------------------

class GatedCNNBlockLTPEv6(nn.Module):
    r""" Our implementation of Gated CNN Block: https://arxiv.org/pdf/1612.08083
    Args: 
        conv_ratio: control the number of channels to conduct depthwise convolution.
            Conduct convolution on partial channels can improve practical efficiency.
            The idea of partial channels is from ShuffleNet V2 (https://arxiv.org/abs/1807.11164) and 
            also used by InceptionNeXt (https://arxiv.org/abs/2303.16900) and FasterNet (https://arxiv.org/abs/2303.03667)
    """
    def __init__(self, dim, expansion_ratio=8/3, kernel_size=7, conv_ratio=1.0,
                 norm_layer=partial(nn.LayerNorm,eps=1e-6), 
                 act_layer=nn.GELU,
                 drop_path=0.,
                 **kwargs):
        super().__init__()
        self.norm = norm_layer(dim)
        hidden = int(expansion_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden * 2)
        self.act = act_layer()
        self.ltpe = Block_line4feature()
        self.fc3 = nn.Linear(hidden, 1)
        self.act2 = act_layer()
        conv_channels = int(conv_ratio * dim)
        self.split_indices = (hidden, hidden - conv_channels, conv_channels) #输出分为三部分：门控信号(g)、直接传递特征(i)和卷积特征(c)
        self.conv = nn.Conv2d(conv_channels, conv_channels, kernel_size=kernel_size, padding=kernel_size//2, groups=conv_channels)
        self.fc2 = nn.Linear(hidden, dim) #将处理后的特征压缩回原始维度
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
    def forward(self, x):
        B, C, H, W = x.shape
        shortcut = x # [B, H, W, C]
        x = self.norm(x)
        g, i, c = torch.split(self.fc1(x), self.split_indices, dim=-1) #通过fc1扩展并分割特征为三部分(g, i, c)
        #print(g.shape,i.shape,c.shape)
        c = c.permute(0, 3, 1, 2) # [B, H, W, C] -> [B, C, H, W]
        c_ltpe = self.ltpe(c)
        c = self.conv(c*c_ltpe)
        c = c.permute(0, 2, 3, 1) # [B, C, H, W] -> [B, H, W, C])
        x = self.fc2(self.act(g) * torch.cat((i, c), dim=-1))  #将g作为门控信号，与i和c的拼接结果相乘
        x = self.drop_path(x)
        return x + shortcut





class StemLayer_LTPE(nn.Module):
    r""" Code modified from InternImage:
        https://github.com/OpenGVLab/InternImage
    """

    def __init__(self,
                 in_channels=3,
                 out_channels=96,
                 act_layer=nn.GELU,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.getltpe = Get_ltpe()
        #self.getltpe = Get_line()
        self.conv1 = nn.Conv2d(in_channels*2,
                               out_channels // 2,
                               kernel_size=3,
                               stride=2,
                               padding=1)
        self.norm1 = norm_layer(out_channels // 2)
        self.act = act_layer()
        self.conv2 = nn.Conv2d(out_channels // 2,
                               out_channels,
                               kernel_size=3,
                               stride=2,
                               padding=1)
        self.norm2 = norm_layer(out_channels)

    def forward(self, x):
        #B, C, H, W = x.shape
        x_ltpe = self.getltpe(x)
        x = torch.cat([x, x_ltpe], dim=1)  # [B, 2*C, H, W]

        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1) # NCHW-> NHWC 归一化层输入形状
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.act(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm2(x)
        return x


DOWNSAMPLE_LAYERS_LTPE = [StemLayer_LTPE] + [DownsampleLayer]*3


class MambaOutLTPE_V6(nn.Module):
    r""" MetaFormer
    using LTPE as extra input
    Our final best model
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3],
                 dims=[96, 192, 384, 576],
                 downsample_layers=DOWNSAMPLE_LAYERS_LTPE,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 act_layer=nn.GELU,
                 conv_ratio=1.0,
                 kernel_size=7,
                 drop_path_rate=0.,
                 output_norm=partial(nn.LayerNorm, eps=1e-6), 
                 head_fn=MlpHead,
                 head_dropout=0.0, 
                 **kwargs,
                 ):
        super().__init__()
        self.num_classes = num_classes

        if not isinstance(depths, (list, tuple)):  
            depths = [depths] # it means the model has only one stage  如果只提供一个值，则将其转换为列表，表示只有一个阶段。
        if not isinstance(dims, (list, tuple)):
            dims = [dims]

        num_stage = len(depths)
        self.num_stage = num_stage

        if not isinstance(downsample_layers, (list, tuple)):
            downsample_layers = [downsample_layers] * num_stage
        down_dims = [in_chans] + dims
        self.downsample_layers = nn.ModuleList(
            [downsample_layers[i](down_dims[i], down_dims[i+1]) for i in range(num_stage)]
        )

        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.stages = nn.ModuleList()
        cur = 0
        for i in range(num_stage):
            stage = nn.Sequential(
                *[GatedCNNBlockLTPEv6(dim=dims[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                kernel_size=kernel_size,
                conv_ratio=conv_ratio,
                drop_path=dp_rates[cur + j],
                ) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = output_norm(dims[-1])

        if head_dropout > 0.0:
            self.head = head_fn(dims[-1], num_classes, head_dropout=head_dropout)
        else:
            self.head = head_fn(dims[-1], num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'norm'}

    def forward_features(self, x):
        for i in range(self.num_stage):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([1, 2])) # (B, H, W, C) -> (B, C)  #x.mean([1, 2])：对特征图进行空间维度的平均池化

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x





###############################################################################
# a series of MambaOut models  
# 这是一个模型注册装饰器，通常用于将模型函数注册到PyTorch的模型库中，允许用户通过torch.hub或模型工厂函数按名称加载模型
@register_model
def mambaout_femto(pretrained=False, **kwargs):
    model = MambaOut(
        depths=[3, 3, 9, 3],
        dims=[48, 96, 192, 288],
        **kwargs)
    model.default_cfg = default_cfgs['mambaout_femto']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url= model.default_cfg['url'], map_location="cpu", check_hash=True) #从 URL 下载并加载预训练权重。
        model.load_state_dict(state_dict)
    return model


@register_model
def mambaout_ltpe(pretrained=False, **kwargs):
    model = MambaOutLTPE_V6(
        depths=[3, 3, 9, 3],
        dims=[48, 96, 192, 288],
        **kwargs)
    #model.default_cfg = default_cfgs['mambaout_femto']
    #if pretrained:
    #    state_dict = torch.hub.load_state_dict_from_url(
    #        url= model.default_cfg['url'], map_location="cpu", check_hash=True) #从 URL 下载并加载预训练权重。
    #    model.load_state_dict(state_dict)
    return model


@register_model
def mambaout_ltpe_V6half(pretrained=False, **kwargs):
    model = MambaOutLTPE_V6(
        depths=[2, 2, 3, 2],
        dims=[48, 96, 192, 288],
        **kwargs)
    #model.default_cfg = default_cfgs['mambaout_femto']
    #if pretrained:
    #    state_dict = torch.hub.load_state_dict_from_url(
    #        url= model.default_cfg['url'], map_location="cpu", check_hash=True) #从 URL 下载并加载预训练权重。
    #    model.load_state_dict(state_dict)
    return model


@register_model
def mambaout_ltpe_V6double(pretrained=False, **kwargs):
    model = MambaOutLTPE_V6(
        depths=[6, 6, 18, 6],
        dims=[48, 96, 192, 288],
        **kwargs)
    #model.default_cfg = default_cfgs['mambaout_femto']
    #if pretrained:
    #    state_dict = torch.hub.load_state_dict_from_url(
    #        url= model.default_cfg['url'], map_location="cpu", check_hash=True) #从 URL 下载并加载预训练权重。
    #    model.load_state_dict(state_dict)
    return model


###############################################################################





# Kobe Memorial Version with 24 Gated CNN blocks
@register_model
def mambaout_kobe(pretrained=False, **kwargs):
    model = MambaOut(
        depths=[3, 3, 15, 3],
        dims=[48, 96, 192, 288],
        **kwargs)
    model.default_cfg = default_cfgs['mambaout_kobe']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url= model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def mambaout_tiny(pretrained=False, **kwargs):
    model = MambaOut(
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 576],
        **kwargs)
    model.default_cfg = default_cfgs['mambaout_tiny']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url= model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def mambaout_small(pretrained=False, **kwargs):
    model = MambaOut(
        depths=[3, 4, 27, 3],
        dims=[96, 192, 384, 576],
        **kwargs)
    model.default_cfg = default_cfgs['mambaout_small']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url= model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def mambaout_base(pretrained=False, **kwargs):
    model = MambaOut(
        depths=[3, 4, 27, 3],
        dims=[128, 256, 512, 768],
        **kwargs)
    model.default_cfg = default_cfgs['mambaout_base']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url= model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model