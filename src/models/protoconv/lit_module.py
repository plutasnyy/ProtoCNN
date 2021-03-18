import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.conv_block import ConvolutionalBlock


class PrototypeLayer(nn.Module):
    def __init__(self, channels_in, number_of_prototypes, kernel_size):
        super().__init__()
        self.prototypes = nn.Parameter(torch.rand([number_of_prototypes, channels_in, kernel_size]), requires_grad=True)
        torch.nn.init.xavier_uniform(self.prototypes.data)

        self.ones = nn.Parameter(torch.ones([number_of_prototypes, channels_in, kernel_size]), requires_grad=False)

    def __call__(self, x):
        x2 = F.conv1d(input=x ** 2, weight=self.ones)
        p2 = torch.sum(self.prototypes ** 2, dim=(1, 2)).view(-1, 1)
        xp = F.conv1d(input=x, weight=self.prototypes)

        distances = F.relu(x2 - 2 * xp + p2)

        return distances


class ProtoConvLitModule(pl.LightningModule):

    def __init__(self, vocab_size, embedding_dim, fold_id, lr, static=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fold_id = fold_id
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.static = static
        self.learning_rate = lr

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1 = ConvolutionalBlock(300, 32, kernel_size=3)
        self.prototype = PrototypeLayer(channels_in=32, number_of_prototypes=32, kernel_size=5)
        self.fc1 = nn.Linear(32, 1, bias=False)

        if self.static:
            self.embedding.weight.requires_grad = False

        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        self.loss = BCEWithLogitsLoss()

    def get_features(self, x):
        x = self.embedding(x).permute((0, 2, 1))
        x = self.conv1(x)
        return x

    def forward(self, x):
        x = self.get_features(x)
        x = self.prototype(x)
        x = -F.max_pool1d(-x, x.size(2))
        x = x.view(x.size(0), -1)
        x = -x # dist to similarity
        logit = self.fc1(x)
        return logit

    def training_step(self, batch, batch_nb):
        outputs = self(batch.text).squeeze(1)
        loss = self.loss(outputs, batch.label)
        preds = torch.round(torch.sigmoid(outputs))

        self.log(f'train_loss_{self.fold_id}', loss, prog_bar=True)
        self.log(f'train_acc_{self.fold_id}', self.train_acc(preds, batch.label), prog_bar=True,
                 on_step=False, on_epoch=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        outputs = self(batch.text).squeeze(1)
        loss = self.loss(outputs, batch.label)
        preds = torch.round(torch.sigmoid(outputs))

        self.log(f'val_loss_{self.fold_id}', loss, prog_bar=True)
        self.log(f'val_acc_{self.fold_id}', self.valid_acc(preds, batch.label), prog_bar=True,
                 on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate, eps=1e-8, weight_decay=0.1)
        return {
            'optimizer': optimizer,
            'lr_scheduler': ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1),
            'monitor': f'val_loss_{self.fold_id}'
        }
