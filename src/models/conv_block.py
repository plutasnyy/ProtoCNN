import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init


class CustomConv1d(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, padding, stride, padding_mode):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.padding = padding
        self.padding_mode = padding_mode

        self.kernel = nn.Parameter(torch.rand([channels_out, channels_in, kernel_size]), requires_grad=True)
        self.bias = nn.Parameter(torch.rand([channels_out]), requires_grad=True)
        self.reset_parameters()

    def forward(self, x):
        if self.padding >= 1:
            x = F.pad(x, (self.padding, self.padding), self.padding_mode)

        batch_size = x.shape[0]
        patches = x.unfold(2, self.kernel_size, self.stride)
        patches = patches.reshape(-1, self.channels_in, self.kernel_size, 1)
        patches = patches * self.kernel.permute(1, 2, 0).unsqueeze(0)  # [1, channels_in, kernel_size, channels_out]
        patches = patches.sum((1, 2)) + self.bias
        patches = patches.reshape(batch_size, self.channels_out, -1)
        return patches

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.kernel, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.kernel)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)


class ConvolutionalBlock(nn.Module):

    def __init__(self, in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1, padding_mode='zeros'):
        super().__init__()
        # self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
        #                        padding_mode=padding_mode)

        self.conv1 = CustomConv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                                  padding_mode=padding_mode)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        return out
