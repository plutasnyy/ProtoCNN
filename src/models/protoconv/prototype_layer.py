import torch
import torch.nn.functional as F
from torch import nn


class PrototypeLayer(nn.Module):
    def __init__(self, channels_in, number_of_prototypes, prototype_length=3, padding=1, padding_mode='constant',
                 initialization='rand'):
        super().__init__()
        self.prototype_length = prototype_length
        self.padding = padding
        self.padding_mode = padding_mode

        if initialization == 'rand':
            self.prototypes = nn.Parameter(torch.rand([number_of_prototypes, channels_in, self.prototype_length]),
                                           requires_grad=True)
        elif initialization == 'zeros':
            self.prototypes = nn.Parameter(torch.zeros([number_of_prototypes, channels_in, self.prototype_length]),
                                           requires_grad=True)
        elif initialization == 'xavier':
            self.prototypes = nn.Parameter(torch.rand([number_of_prototypes, channels_in, self.prototype_length]),
                                           requires_grad=True)
            torch.nn.init.xavier_uniform(self.prototypes.data)
        else:
            raise KeyError(f'Invalid initialization parameter {initialization}, '
                           f'only ["rand", "zeros", "xavier"] are allowed')
        self.ones = nn.Parameter(torch.ones([number_of_prototypes, channels_in, self.prototype_length]),
                                 requires_grad=False)

    def __call__(self, x):
        if self.padding >= 1:
            x = F.pad(x, (self.padding, self.padding), self.padding_mode)

        x2 = F.conv1d(input=x ** 2, weight=self.ones)
        xp = F.conv1d(input=x, weight=self.prototypes)
        p2 = torch.sum(self.prototypes ** 2, dim=(1, 2)).view(-1, 1)

        distances = F.relu(x2 - 2 * xp + p2)

        return distances


if __name__ == "__main__":
    batch_size = 32
    length = 20
    emb_dim = 300

    input = torch.rand([batch_size, emb_dim, length])
    pl = PrototypeLayer(300, number_of_prototypes=50, prototype_length=5, padding=2, padding_mode='constant')

    x = pl(input)

    print(x.shape)
    print(x[1, 3, 10])  # 1 exmaple, 3 prototype, 10th word
    print(torch.norm(input[1, :, 8:13] - pl.prototypes[3, :, :], p=2) ** 2)
