#Basic Modules

#Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class SE(nn.Module):
    def __init__(self, in_channels, reduction_ratio):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels//reduction_ratio, bias = True)
        self.relu = nn.ReLU(inplace = True)
        self.fc2 = nn.Linear(in_channels//reduction_ratio, in_channels, bias = True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        out = self.pool(x).view(b, c)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out).view(b, c, 1, 1)
        return x * out.expand_as(x)

class SRM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.cfc = Parameter(torch.Tensor(in_channels, 2))
        self.cfc.data.fill_(0)
        self.bn = nn.BatchNorm2d(in_channels)
        self.activation = nn.Sigmoid()

    def _style_pooling(self, x, eps=1e-5):
        N, C, _, _ = x.size()
        channel_mean = x.view(N, C, -1).mean(dim=2, keepdim=True)
        channel_var = x.view(N, C, -1).var(dim=2, keepdim=True) + eps
        channel_std = channel_var.sqrt()
        t = torch.cat((channel_mean, channel_std), dim=2)
        return t 
    
    def _style_integration(self, t):
        z = t * self.cfc[None, :, :]  
        z = torch.sum(z, dim=2)[:, :, None, None] 
        z_hat = self.bn(z)
        g = self.activation(z_hat)
        return g

    def forward(self, x):
        t = self._style_pooling(x)
        g = self._style_integration(t)
        return x * g

class BAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio = 4):
        super().__init__()
        #Channel Attention
        self.c_pool = nn.AdaptiveAvgPool2d(1)
        self.c_fc1 = nn.Linear(in_channels, in_channels // reduction_ratio, bias = True)
        self.c_fc2 = nn.Linear(in_channels // reduction_ratio, in_channels, bias = True)
        self.c_bn = nn.BatchNorm2d(in_channels)

        #Spatial Attention
        self.s_conv1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size = 1)
        self.s_bn1 = nn.BatchNorm2d(in_channels // reduction_ratio)
        self.s_relu1 = nn.ReLU()
        self.s_conv2 = nn.Conv2d(in_channels // reduction_ratio, in_channels // reduction_ratio, kernel_size = 3, padding = 4, dilation = 4)
        self.s_bn2 = nn.BatchNorm2d(in_channels // reduction_ratio)
        self.s_relu2 = nn.ReLU() 
        self.s_conv3 = nn.Conv2d(in_channels // reduction_ratio, in_channels // reduction_ratio, kernel_size = 3, padding = 4, dilation = 4)
        self.s_bn3 = nn.BatchNorm2d(in_channels // reduction_ratio)
        self.s_relu3 = nn.ReLU()
        self.s_conv4 = nn.Conv2d(in_channels // reduction_ratio, 1, kernel_size = 1)

        #Combine
        self.sigmoid = nn.Sigmoid()

    def channel_attention(self, x):
        b, c, _, _ = x.size()
        out = self.c_pool(x).view(b, c)
        out = self.c_fc1(out)
        out = self.c_fc2(out).view(b, c, 1, 1)
        out = self.c_bn(out)
        return out
    
    def spatial_attention(self, x):
        out = self.s_conv1(x)
        out = self.s_bn1(out)
        out = self.s_relu1(out)
        out = self.s_conv2(out)
        out = self.s_bn2(out)
        out = self.s_relu2(out)
        out = self.s_conv3(out)
        out = self.s_bn3(out)
        out = self.s_relu3(out)
        out = self.s_conv4(out)
        return out
    
    def forward(self, x):
        m_c = self.channel_attention(x)
        m_s = self.spatial_attention(x)
        out = self.sigmoid(m_c * m_s)
        return x * out

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio = 4):
        super().__init__()

        #Channel Attention(MaxPool)
        self.c_pool1 = nn.AdaptiveAvgPool2d(1)
        self.c_fc1 = nn.Linear(in_channels, in_channels // reduction_ratio, bias = True)
        self.c_relu1 = nn.ReLU()
        self.c_fc2 = nn.Linear(in_channels // reduction_ratio, in_channels, bias = True)

        #Channel Attention(AvgPool)
        self.c_pool2 = nn.AdaptiveMaxPool2d(1)
        self.c_fc3 = nn.Linear(in_channels, in_channels // reduction_ratio, bias = True)
        self.c_relu2 = nn.ReLU()
        self.c_fc4 = nn.Linear(in_channels // reduction_ratio, in_channels, bias = True)

        #Combine
        self.c_sigmoid1 = nn.Sigmoid()

        #Spatial Attention
        self.s_conv1 = nn.Conv2d(2, 1, kernel_size = 7, stride = 1, padding = 3)
        self.s_bn1 = nn.BatchNorm2d(1)
        self.s_sigmoid2 = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()

        out_max = self.c_pool1(x).view(b, c)
        out_max = self.c_fc1(out_max)
        out_max = self.c_relu1(out_max)
        out_max = self.c_fc2(out_max).view(b, c, 1, 1)

        out_avg = self.c_pool2(x).view(b, c)
        out_avg = self.c_fc3(out_avg)
        out_avg = self.c_relu2(out_avg)
        out_avg = self.c_fc4(out_avg).view(b, c, 1, 1)

        out_channel = self.c_sigmoid1(out_max + out_avg)
        out_channel = x * out_channel

        out_spatial = torch.cat((torch.max(out_channel,1)[0].unsqueeze(1), torch.mean(out_channel,1).unsqueeze(1)), dim=1)
        out_spatial = self.s_conv1(out_spatial)
        out_spatial = self.s_bn1(out_spatial)
        out_spatial = self.s_sigmoid2(out_spatial)
        out_spatial = out_channel * out_spatial
        return out_spatial
        
class GE(nn.Module):
    def __init__(self, in_channels, location):
        super().__init__()
        kernel_size = [56, 28, 14, 7]
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size = kernel_size[location], groups = in_channels)
        self.bn = nn.BatchNorm2d(in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        out = self.conv(x)
        out = self.bn(out)
        out = self.sigmoid(out)
        return x * out

class SepConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size = kernel_size, stride = 1, padding = 1, groups = in_channels, bias = False)
        self.bn1 = nn.BatchNorm2d(in_channels, eps = 1e-05, momentum = 0.01)
        self.relu1 = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels, eps = 1e-05, momentum = 0.01)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return out

class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, expansion_ratio, residual = True, attention = None):
        super().__init__()
        self.attention = attention
        self.residual = residual
        expanded_channels = in_channels * expansion_ratio
        if self.residual == True:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride, bias = False)
        self.conv1 = nn.Conv2d(in_channels, expanded_channels, kernel_size = 1, stride = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(expanded_channels, eps = 1e-05, momentum = 0.01)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(expanded_channels, expanded_channels, kernel_size = kernel_size, stride = stride, padding = padding, groups = expanded_channels, bias = False)
        self.bn2 = nn.BatchNorm2d(expanded_channels, eps = 1e-05, momentum = 0.01)
        self.relu2 = nn.ReLU()
        if self.attention == "SE":
            self.se = SEBlock(expanded_channels, reduction_ratio = 4)
        self.conv3 = nn.Conv2d(expanded_channels, out_channels, kernel_size = 1, stride = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(out_channels, eps = 1e-05, momentum = 0.01)

    def forward(self, x):
        if self.residual == True:
            residual = self.residual(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        if self.attention == "SE":
            out = self.se(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.residual == True:
            out += residual
        return out

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.stride = stride
        if self.stride != 1:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride, bias = False)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        residual = x
        if self.stride != 1:
            residual = self.residual(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out += residual
        return out

class Bottleneck_Block(nn.Module):
    def __init__(self, in_channels, bottleneck_channels, out_channels, stride):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if self.in_channels != self.out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride, bias = False)
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size = 1, stride = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size = 1, stride = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3= nn.ReLU()

    def forward(self, x):
        residual = x
        if self.in_channels != self.out_channels:
            residual = self.residual(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out += residual
        return out

class Depthwise_Separable_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size = 3, stride = stride, padding = 1, groups = in_channels, bias = False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        return out


    