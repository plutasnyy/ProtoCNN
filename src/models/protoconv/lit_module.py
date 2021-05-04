import math
from copy import copy

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import BCEWithLogitsLoss
from torch.nn.init import calculate_gain
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.conv_block import ConvolutionalBlock
from models.embeddings_dataset_utils import get_dataset
from models.protoconv.prototype_layer import PrototypeLayer
from models.protoconv.prototype_projection import PrototypeProjection
from models.protoconv.return_wrappers import LossesWrapper, PrototypeDetailPrediction


class ProtoConvLitModule(pl.LightningModule):
    dist_to_sim = {
        'linear': lambda x: -x,
        'log': lambda x: torch.log((x + 1) / (x + 1e-4))
    }

    def __init__(self, vocab_size, embedding_dim, fold_id=1, lr=1e-3, static_embedding=True,
                 pc_project_prototypes_every_n=4, pc_sim_func='log', pc_separation_threshold=10,
                 pc_number_of_prototypes=16, pc_conv_filters=32, pc_ce_loss_weight=1, pc_sep_loss_weight=0,
                 pc_cls_loss_weight=0, pc_l1_loss_weight=0, pc_conv_stride=1, pc_conv_filter_size=3, pc_conv_padding=1,
                 pc_prototypes_init='rand', vocab_itos=None, *args, **kwargs):
        super().__init__()

        self.save_hyperparameters()

        self.fold_id = fold_id
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.learning_rate = lr
        self.project_prototypes_every_n = pc_project_prototypes_every_n
        self.sim_func = pc_sim_func
        self.separation_threshold = pc_separation_threshold
        self.ce_loss_weight = pc_ce_loss_weight
        self.sep_loss_weight = pc_sep_loss_weight
        self.cls_loss_weight = pc_cls_loss_weight
        self.l1_loss_weight = pc_l1_loss_weight
        self.number_of_prototypes: int = pc_number_of_prototypes
        self.conv_filters: int = pc_conv_filters
        self.conv_stride: int = pc_conv_stride
        self.conv_filter_size = pc_conv_filter_size
        self.conv_padding = pc_conv_padding
        self.prototypes_init = pc_prototypes_init

        self.max_number_of_prototypes = 100
        self.current_prototypes_number = self.number_of_prototypes
        self.enabled_prototypes_mask = nn.Parameter(torch.cat([
            torch.ones(self.current_prototypes_number),
            torch.zeros(self.max_number_of_prototypes - self.current_prototypes_number)
        ]), requires_grad=False)

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1 = ConvolutionalBlock(300, self.conv_filters, kernel_size=self.conv_filter_size,
                                        padding=self.conv_padding, stride=self.conv_stride, padding_mode="reflect")
        self.prototypes = PrototypeLayer(channels_in=self.conv_filters,
                                         number_of_prototypes=self.max_number_of_prototypes,
                                         initialization=self.prototypes_init)
        self.fc1 = nn.Linear(self.max_number_of_prototypes, 1, bias=False)
        self._init_fc_layer()
        self._zeroing_disabled_prototypes()

        self.prototype_projection: PrototypeProjection = PrototypeProjection(self.prototypes.prototypes.shape)

        if static_embedding:
            self.embedding.weight.requires_grad = False

        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        self.loss = BCEWithLogitsLoss()

        self.last_train_losses = None
        self.vocab_itos = vocab_itos

    def get_features(self, x):
        x = self.embedding(x).permute((0, 2, 1))
        x = self.conv1(x)
        return x

    def forward(self, x):
        latent_space = self.get_features(x)
        distances = self.prototypes(latent_space)
        min_dist = self._min_pooling(distances)
        similarity = self.dist_to_sim[self.sim_func](min_dist)
        masked_similarity = similarity * self.enabled_prototypes_mask
        logits = self.fc1(masked_similarity).squeeze(1)
        return PrototypeDetailPrediction(latent_space, distances, logits, min_dist)

    def on_train_epoch_start(self, *args, **kwargs):
        if self._is_projection_prototype_epoch():
            self.prototype_projection.reset(device=self.device)

    def training_step(self, batch, batch_nb):
        if self._is_projection_prototype_epoch():
            self.project_prototypes(batch)
            loss = torch.tensor(.0, requires_grad=True)
            losses = LossesWrapper(loss, 0, 0, 0, 0, 0) if self.last_train_losses is None else self.last_train_losses
        else:
            losses = self.learning_step(batch, self.train_acc)
            loss = losses.loss
            self.last_train_losses = losses

        self.log_all_metrics('train', losses)
        return {'loss': loss}

    def project_prototypes(self, batch):
        self.eval()
        with torch.no_grad():
            prediction: PrototypeDetailPrediction = self(batch.text)
            self.prototype_projection.update(prediction)

    def on_train_epoch_end(self, *args, **kwargs):
        if self._is_projection_prototype_epoch():
            self.prototypes.prototypes.data.copy_(self.prototype_projection.get_weights())
            self._zeroing_disabled_prototypes()
            print('The prototypes were projected')

    def validation_step(self, batch, batch_nb):
        losses = self.learning_step(batch, self.valid_acc)
        self.log_all_metrics('val', losses)

    def learning_step(self, batch, acc_score):
        outputs = self(batch.text)
        preds = torch.round(torch.sigmoid(outputs.logits))

        cross_entropy = self.loss(outputs.logits, batch.label)
        clustering_loss = self.calculate_clustering_loss(outputs)
        separation_loss = self.calculate_separation_loss(self.prototypes.prototypes,
                                                         threshold=self.separation_threshold)
        l1 = self.fc1.weight.norm(p=1)
        loss = self.ce_loss_weight * cross_entropy + self.cls_loss_weight * clustering_loss + \
               self.sep_loss_weight * separation_loss + self.l1_loss_weight * l1
        accuracy = acc_score(preds, batch.label)

        return LossesWrapper(loss, cross_entropy, clustering_loss, separation_loss, l1, accuracy)

    def log_all_metrics(self, stage, losses: LossesWrapper):
        self.log(f'{stage}_loss_{self.fold_id}', losses.loss, prog_bar=True)
        self.log(f'{stage}_ce_{self.fold_id}', losses.cross_entropy, prog_bar=False)
        self.log(f'{stage}_clst_{self.fold_id}', losses.clustering_loss, prog_bar=False)
        self.log(f'{stage}_sep_{self.fold_id}', losses.separation_loss, prog_bar=False)
        self.log(f'{stage}_l1_{self.fold_id}', losses.l1, prog_bar=False)
        self.log(f'{stage}_acc_{self.fold_id}', losses.accuracy, prog_bar=True, on_step=False, on_epoch=True)

    @staticmethod
    def _min_pooling(x):
        x = -F.max_pool1d(-x, x.size(2))
        x = x.view(x.size(0), -1)
        return x

    @staticmethod
    def calculate_separation_loss(prototypes, threshold):  # [prototypes, latent_size, 1]
        """
        :param prototypes:
        :param threshold: the threshold, after that higher distances are ignored: distance = max(threshold-separation,0)
        """
        # TODO distance is not squared, clustering loss uses squared distances
        prot = prototypes.squeeze(2).unsqueeze(0)  # [1, prototypes, latent_size]
        distances_matrix = torch.cdist(prot, prot, p=2).squeeze(0)  # [prototypes, prototypes]
        max_value = torch.max(distances_matrix) + 1
        distances_matrix_no_zeros = distances_matrix + torch.eye(prototypes.shape[0]).to(prototypes.device) * max_value
        min_distances, _ = torch.min(distances_matrix_no_zeros, dim=1)
        mean_separation_distance = torch.mean(min_distances)
        loss = F.relu(threshold - mean_separation_distance)
        return loss

    @staticmethod
    def calculate_clustering_loss(outputs: PrototypeDetailPrediction):
        cluster_cost = torch.mean(outputs.min_distances)
        return cluster_cost

    def _is_projection_prototype_epoch(self):
        return self.project_prototypes_every_n > 0 and (self.current_epoch + 1) % self.project_prototypes_every_n == 0

    def _remove_prototypes(self, prototype_ids, target_prototype_ids=None):
        """
        :param prototype_ids: list IDs of prototypes to remove
        :param target_prototype_ids: When the 2 prototypes are almost identical, you don't just want to delete
        the prototype, but also transfer its weight to the other prototype. target_prototype_id is used to specify
        the location of the prototype where the weight from prototype_id will be transferred. The result of softmax
        should remain unchanged (as long as the prototypes represented the classes in the same way). If the prototype
        is completely irrelevant (e.g. its weight in the FC layer is 0), simply remove it and let this parameter be None
        """
        with torch.no_grad():
            zero_indices = torch.nonzero(self.enabled_prototypes_mask == 0, as_tuple=False).squeeze(1).cpu().numpy()

            assert len(set(zero_indices) & set(prototype_ids)) == 0, \
                f'You are trying to remove prototypes that dont exist: {set(zero_indices) & set(prototype_ids)}'

            if target_prototype_ids is not None:
                assert len(set(zero_indices) & set(target_prototype_ids)) == 0, \
                    f'You are trying to add value to prototypes that dont exist: {set(zero_indices) & set(prototype_ids)}'
                self.fc1.weight.data[0, target_prototype_ids] += self.fc1.weight.data[0, prototype_ids]

            self.enabled_prototypes_mask[prototype_ids] = 0

        self._zeroing_disabled_prototypes()
        print(f'Prototypes {prototype_ids} were removed')

    def _add_prototype(self):
        if self.current_prototypes_number < self.max_number_of_prototypes:
            with torch.no_grad():
                zero_indices = torch.nonzero(self.enabled_prototypes_mask == 0, as_tuple=False).squeeze(1)
                new_prototype_id = zero_indices[0].item()

                std = calculate_gain('leaky_relu', math.sqrt(5)) / math.sqrt(self.current_prototypes_number + 1)
                bound = math.sqrt(3.0) * std
                self.fc1.weight.data[0, new_prototype_id].uniform_(-bound, bound)
                self.prototypes.prototypes.data[new_prototype_id].uniform_(0, 1)
            self.current_prototypes_number += 1
            print(f'Added new prototype, current number of prototypes: {self.prototypes.prototypes.shape[0]}')

    def _init_fc_layer(self):
        std = calculate_gain('leaky_relu', math.sqrt(5)) / math.sqrt(self.current_prototypes_number)
        bound = math.sqrt(3.0) * std
        with torch.no_grad():
            self.fc1.weight.uniform_(-bound, bound)  # kaiming_uniform_

    def _zeroing_disabled_prototypes(self):
        with torch.no_grad():
            self.prototypes.prototypes.data[~self.enabled_prototypes_mask.bool()] *= 0
            self.fc1.weight.data[:, ~self.enabled_prototypes_mask.bool()] *= 0

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate, eps=1e-8, weight_decay=0.1)
        lr_scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1, verbose=True),
            'name': f'learning_rate_{self.fold_id}',
            'monitor': f'val_loss_{self.fold_id}'
        }
        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
        }

    @staticmethod
    def calc_number_of_epochs_with_projection(epochs, period):
        """
        prototype projection is done as 'dry' epoch, so the number of max_epochs should be increased
        """
        if period <= 1:
            return epochs
        epochs_to_do, epoch_iterator = copy(epochs), 0
        while epochs_to_do > 0:
            if (epoch_iterator + 1) % period != 0:
                epochs_to_do -= 1
            epoch_iterator += 1
        return epoch_iterator

    @classmethod
    def from_params_and_dataset(cls, train_df, valid_df, params, fold_id, embeddings=None):
        TEXT, LABEL, train_loader, val_loader = get_dataset(train_df, valid_df, params.batch_size, gpus=params.gpu,
                                                            vectors=embeddings)
        itos = TEXT.vocab.itos if params.pc_visualize else None
        model = cls(vocab_size=len(TEXT.vocab), embedding_dim=TEXT.vocab.vectors.shape[1], fold_id=fold_id,
                    vocab_itos=itos, **params)
        model.embedding.weight.data.copy_(TEXT.vocab.vectors)
        return model, train_loader, val_loader
