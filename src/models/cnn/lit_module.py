import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR


class CNNLitModule(pl.LightningModule):

    def __init__(self, vocab_size, embedding_length, fold_id, lr, static=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fold_id = fold_id
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length
        self.static = static
        self.learning_rate = lr

        V = vocab_size
        D = embedding_length
        C = 1
        Ci = 1
        Co = 32  # kernel number
        Ks = [3, 5, 7, 10]  # kernel sizes

        self.embedding = nn.Embedding(V, D)
        self.convs = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(len(Ks) * Co, C)

        if self.static:
            self.embedding.weight.requires_grad = False

        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        self.loss = BCEWithLogitsLoss()

    def forward(self, x):
        x = self.embedding(x)  # (N, W, D)
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x).squeeze(1)  # (N, C)
        return logit

    def training_step(self, batch, batch_nb):
        outputs = self(batch.text)
        loss = self.loss(outputs, batch.label)
        preds = torch.round(torch.sigmoid(outputs))

        self.log(f'train_loss_{self.fold_id}', loss, prog_bar=True)
        self.log(f'train_acc_{self.fold_id}', self.train_acc(preds, batch.label), prog_bar=True,
                 on_step=False, on_epoch=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        outputs = self(batch.text)
        loss = self.loss(outputs, batch.label)
        preds = torch.round(torch.sigmoid(outputs))

        self.log(f'val_loss_{self.fold_id}', loss, prog_bar=True)
        self.log(f'val_acc_{self.fold_id}', self.valid_acc(preds, batch.label), prog_bar=True,
                 on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate, eps=1e-8)
        return {
            'optimizer': optimizer,
            'lr_scheduler': StepLR(optimizer, step_size=1, gamma=0.1),
            'monitor': f'val_loss_{self.fold_id}'
        }
