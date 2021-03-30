import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.conv_block import ConvolutionalBlock


class CNNLitModule(pl.LightningModule):

    def __init__(self, vocab_size, embedding_dim, fold_id, lr, static=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        self.fold_id = fold_id
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.static = static
        self.learning_rate = lr

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1 = ConvolutionalBlock(300, 32, kernel_size=3)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(32, 1)

        if self.static:
            self.embedding.weight.requires_grad = False

        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        self.loss = BCEWithLogitsLoss()

    def forward(self, x):
        x = self.embedding(x).permute((0, 2, 1))
        x = self.conv1(x)
        x = F.max_pool1d(x, x.size(2))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
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
