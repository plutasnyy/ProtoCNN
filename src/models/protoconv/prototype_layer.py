import torch
import torch.nn.functional as F
from torch import nn


class PrototypeLayer(nn.Module):
    def __init__(self, channels_in, number_of_prototypes):
        super().__init__()
        self.prototypes = nn.Parameter(torch.rand([number_of_prototypes, channels_in, 1]), requires_grad=True)
        torch.nn.init.xavier_uniform(self.prototypes.data)
        self.ones = nn.Parameter(torch.ones([number_of_prototypes, channels_in, 1]), requires_grad=False)

    def __call__(self, x):
        x2 = F.conv1d(input=x ** 2, weight=self.ones)
        xp = F.conv1d(input=x, weight=self.prototypes)
        p2 = torch.sum(self.prototypes ** 2, dim=1).view(-1, 1)

        distances = F.relu(x2 - 2 * xp + p2)

        return distances
