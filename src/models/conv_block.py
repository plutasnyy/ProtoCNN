import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class CustomConv1d(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, padding, stride, padding_mode):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.padding = padding
        self.padding_mode = padding_mode

        self.weight = nn.Parameter(torch.rand([channels_out, channels_in, kernel_size]), requires_grad=True)
        self.words_regularization = nn.Parameter(torch.ones([channels_out, kernel_size]), requires_grad=False)
        self.bias = nn.Parameter(torch.rand([channels_out]), requires_grad=True)

    def forward(self, x):
        if self.padding >= 1:
            x = F.pad(x, (self.padding, self.padding), self.padding_mode) # [batch_size, in_channels, words]

        batch_size = x.shape[0]
        patches = x.unfold(2, self.kernel_size, self.stride) # [batch_size, in_channels, windows, kernel_size]
        patches = patches.permute(0, 2, 1, 3).reshape(-1, self.channels_in, self.kernel_size) # [batch_size x windows, channels_in, kernel_size]
        patches = patches.unsqueeze(3) * self.weight.permute(1, 2, 0).unsqueeze(0)        # [batch_size x windows, channels_in, kernel_size, 1] x [1, channels_in , kernel_size , filters] = [batch_size x windows, channels_in, kernel_size, channels_out]
        patches = patches.sum(1) * self.words_regularization.permute(1,0)  # [batch_size x windows, kernel_size, channels_out]
        patches = patches.sum(1) + self.bias # [batch_size x windows, channels_out]
        patches = patches.reshape(batch_size, -1, self.channels_out).permute(0, 2, 1) # [batch size, channels_out, windows]
        return patches


class ConvolutionalBlock(nn.Module):

    def __init__(self, in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1, padding_mode='zeros'):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                               padding_mode=padding_mode)

        # self.conv1 = CustomConv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
        #                           padding_mode=padding_mode)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        return out


if __name__ == '__main__':
    in_channels = 3
    out_channels = 5
    kernel_size = 2
    padding = 0
    stride = 1
    padding_mode = 'reflect'

    conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                       padding_mode=padding_mode)
    conv1d_2 = CustomConv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                            padding_mode=padding_mode)

    text = torch.tensor(np.array([
        [[1, 0, 0.2, 0], [2, 0, 0, 0], [3, 3, 0, 0.9]],
        [[1, 0, 0, -1], [0, 1, 0, -2], [0, 0, 0, -3]]
    ])).float()
    conv1d_2.weight.data.copy_(conv1d.weight.data)
    conv1d.bias.data.copy_(conv1d_2.bias.data)

    print(conv1d(text))
    print(conv1d_2(text))
