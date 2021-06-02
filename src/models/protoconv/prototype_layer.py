import torch
import torch.nn.functional as F
from torch import nn


class PrototypeLayer(nn.Module):
    def __init__(self, channels_in, number_of_prototypes, initialization='rand'):
        super().__init__()
        if initialization == 'rand':
            self.prototypes = nn.Parameter(torch.rand([number_of_prototypes, channels_in, 1]), requires_grad=True)
        elif initialization == 'zeros':
            self.prototypes = nn.Parameter(torch.zeros([number_of_prototypes, channels_in, 1]), requires_grad=True)
        elif initialization == 'xavier':
            self.prototypes = nn.Parameter(torch.rand([number_of_prototypes, channels_in, 1]), requires_grad=True)
            torch.nn.init.xavier_uniform(self.prototypes.data)
        else:
            raise KeyError(f'Invalid initialization parameter {initialization}, '
                           f'only ["rand", "zeros", "xavier"] are allowed')
        self.ones = nn.Parameter(torch.ones([number_of_prototypes, channels_in, 1]), requires_grad=False)

    def __call__(self, x):
        x2 = F.conv1d(input=x ** 2, weight=self.ones)
        xp = F.conv1d(input=x, weight=self.prototypes)
        p2 = torch.sum(self.prototypes ** 2, dim=1).view(-1, 1)

        distances = F.relu(x2 - 2 * xp + p2)

        return distances


if __name__ == "__main__":
    batch_size = 32
    emb_dim = 300
    length = 20

    input = torch.rand([batch_size, emb_dim, length])
    pl = PrototypeLayer(300, number_of_prototypes=50)
    x = pl(input)

    print(x.shape)
    print(x[1, 3, 10])  # 1 exmaple, 3 prototype, 10th word
    print(torch.norm(input[1, :, 10] - pl.prototypes[3, :, 0], p=2) ** 2)
