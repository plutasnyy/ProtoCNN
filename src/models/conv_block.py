from torch import nn


class ConvolutionalBlock(nn.Module):

    def __init__(self, in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1, padding_mode='zeros'):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                               padding_mode=padding_mode)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        return out
