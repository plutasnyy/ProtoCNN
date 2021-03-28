from collections import namedtuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.conv_block import ConvolutionalBlock
from models.protoconv.prototype_layer import PrototypeLayer


class ProtoConvLitModule(pl.LightningModule):

    def __init__(self, vocab_size, embedding_dim, fold_id=1, lr=1e-3, static_embedding=True,
                 project_prototypes_every_n=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fold_id = fold_id
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.static = static_embedding
        self.learning_rate = lr
        self.project_prototypes_every_n = project_prototypes_every_n

        self.number_of_prototypes: int = 16
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1 = ConvolutionalBlock(300, 32, kernel_size=3, padding=1)
        self.prototypes = PrototypeLayer(channels_in=32, number_of_prototypes=self.number_of_prototypes)
        self.fc1 = nn.Linear(16, 1, bias=False)

        if self.static:
            self.embedding.weight.requires_grad = False

        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        self.loss = BCEWithLogitsLoss()

        self.projected_prototypes = torch.zeros_like(self.prototypes.prototypes)
        self.min_distances_prototype_example = torch.full([self.number_of_prototypes], fill_value=float('inf'))
        self.min_distances_prototype_example[0] = float('-inf')

    def get_features(self, x):
        x = self.embedding(x).permute((0, 2, 1))
        x = self.conv1(x)
        return x

    def forward(self, x, get_representation=False):
        latent_space = self.get_features(x)
        distances = self.prototypes(latent_space)
        min_dist = self._min_pooling(distances)
        similarity = -min_dist  # dist to similarity
        logits = self.fc1(similarity).squeeze(1)

        if get_representation:
            PrototypeDetailPrediction = namedtuple('PrototypeDetailPrediction', 'latent_space distances')
            return PrototypeDetailPrediction(latent_space, distances)
        return logits

    def training_step(self, batch, batch_nb):
        if self._is_projection_prototype_epoch():
            self.project_prototypes(batch)
            loss = torch.tensor(.0, requires_grad=True)
        else:
            outputs = self(batch.text)
            loss = self.loss(outputs, batch.label)
            preds = torch.round(torch.sigmoid(outputs))

            self.log(f'train_loss_{self.fold_id}', loss, prog_bar=True)
            self.log(f'train_acc_{self.fold_id}', self.train_acc(preds, batch.label), prog_bar=True,
                     on_step=False, on_epoch=True)
        return {'loss': loss}

    def project_prototypes(self, batch):
        # TODO consider only with specific class
        # TODO consider removing the same prototypes

        self.eval()
        with torch.no_grad():
            prediction = self(batch.text, get_representation=True)
            for pred_id, label in enumerate(batch.label):
                latent_space = prediction.latent_space[pred_id].squeeze(0)  # [32,115] [ latent_size, length]
                distances = prediction.distances[pred_id].squeeze(0)  # [16,115]  [prototypes, distances]

                assert distances.shape[1] == latent_space.shape[1]
                best_distances_idx = distances.argmin(dim=1)
                best_distances = distances[torch.arange(distances.shape[0]), best_distances_idx]
                best_latents_to_prototype = latent_space.permute(1, 0)[best_distances_idx].unsqueeze(2)
                # permute to [115, 32] for correct selecting representations, after selection [prototypes, 32, 1]

                update_mask = best_distances < self.min_distances_prototype_example
                update_indexes = torch.where(update_mask == 1)

                self.min_distances_prototype_example[update_indexes] = best_distances[update_indexes]
                self.projected_prototypes[update_indexes] = best_latents_to_prototype[update_indexes]  # [16,32,1]

    def training_epoch_end(self, *args, **kwargs):
        if self._is_projection_prototype_epoch():
            self.prototypes.prototypes.data.copy_(torch.tensor(self.projected_prototypes))
            self.projected_prototypes = torch.zeros_like(self.prototypes.prototypes)
            self.min_distances_prototype_example = torch.full([self.number_of_prototypes], float('inf'))

    def validation_step(self, batch, batch_nb):
        outputs = self(batch.text)
        loss = self.loss(outputs, batch.label)
        preds = torch.round(torch.sigmoid(outputs))

        self.log(f'val_loss_{self.fold_id}', loss, prog_bar=True)
        self.log(f'val_acc_{self.fold_id}', self.valid_acc(preds, batch.label), prog_bar=True, on_step=False,
                 on_epoch=True)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate, eps=1e-8, weight_decay=0.1)
        return {
            'optimizer': optimizer,
            'lr_scheduler': ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1),
            'monitor': f'val_loss_{self.fold_id}'
        }

    def _is_projection_prototype_epoch(self):
        return (self.current_epoch + 1) % self.project_prototypes_every_n == 0

    @staticmethod
    def _min_pooling(x):
        x = -F.max_pool1d(-x, x.size(2))
        x = x.view(x.size(0), -1)
        return x
