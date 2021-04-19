import math

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import BCEWithLogitsLoss
from torch.nn.init import _no_grad_normal_
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
                 pc_number_of_prototypes=16, pc_conv_filters=32, pc_sep_loss_weight=0, pc_cls_loss_weight=0,
                 pc_stride=1, pc_filter_size=3, pc_prototypes_init='rand', vocab_itos=None, *args, **kwargs):
        super().__init__()

        self.save_hyperparameters()

        self.fold_id = fold_id
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.learning_rate = lr
        self.project_prototypes_every_n = pc_project_prototypes_every_n
        self.sim_func = pc_sim_func
        self.separation_threshold = pc_separation_threshold
        self.sep_loss_weight = pc_sep_loss_weight
        self.cls_loss_weight = pc_cls_loss_weight
        self.number_of_prototypes: int = pc_number_of_prototypes
        self.conv_filters: int = pc_conv_filters
        self.conv_stride: int = pc_stride
        self.filter_size = pc_filter_size
        self.prototypes_init = pc_prototypes_init

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1 = ConvolutionalBlock(300, self.conv_filters, kernel_size=self.filter_size, padding=0,
                                        stride=self.conv_stride, padding_mode="reflect")
        self.prototypes = PrototypeLayer(channels_in=self.conv_filters, number_of_prototypes=self.number_of_prototypes,
                                         initialization=self.prototypes_init)
        self.fc1 = nn.Linear(self.number_of_prototypes, 1, bias=False)
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
        logits = self.fc1(similarity).squeeze(1)
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

        # self._remove_prototype(0)
        # self._add_prototype()
        #
        # _, ids = torch.where(torch.abs(self.fc1.weight) < 0.001)
        # print(f'Remove prototypes: {ids.cpu().numpy()}')
        # for shift, i in enumerate(ids.cpu().numpy()):
        #     # the prototype ids are shifted after removing each one
        #     self._remove_prototype(i - shift)

        # self._add_prototype()

    def validation_step(self, batch, batch_nb):
        losses = self.learning_step(batch, self.valid_acc)
        self.log_all_metrics('val', losses)

    def learning_step(self, batch, acc_score):
        outputs = self(batch.text)
        preds = torch.round(torch.sigmoid(outputs.logits))

        cross_entropy = self.loss(outputs.logits, batch.label)
        clustering_loss = self.calculate_clustering_loss(outputs)
        separation_loss = self.calculate_separation_loss(self.prototypes.prototypes, alpha=self.separation_threshold)
        l1 = self.fc1.weight.norm(p=1)
        loss = cross_entropy + self.cls_loss_weight * clustering_loss + self.sep_loss_weight * separation_loss + 1e-2 * l1
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
    def calculate_separation_loss(prototypes, alpha):  # [prototypes, latent_size, 1]
        """
        :param prototypes:
        :param alpha: the threshold, after that higher distances are ignored: distance = max(distance,alpha)
        """
        # TODO distance is not squared, clustering loss uses squared distances
        prot = prototypes.squeeze(2).unsqueeze(0)  # [1, prototypes, latent_size]
        distances_matrix = torch.cdist(prot, prot, p=2).squeeze(0)  # [prototypes, prototypes]
        max_value = torch.max(distances_matrix) + 1
        distances_matrix_no_zeros = distances_matrix + torch.eye(prototypes.shape[0]).to(prototypes.device) * max_value
        min_distances, _ = torch.min(distances_matrix_no_zeros, dim=1)
        mean_separation_distance = torch.mean(min_distances)
        clip_dist = torch.clip(mean_separation_distance, max=alpha)
        loss = -clip_dist
        return loss

    @staticmethod
    def calculate_clustering_loss(outputs: PrototypeDetailPrediction):
        cluster_cost = torch.mean(outputs.min_distances)
        return cluster_cost

    def _is_projection_prototype_epoch(self):
        return self.project_prototypes_every_n > 0 and (self.current_epoch + 1) % self.project_prototypes_every_n == 0

    def _remove_prototype(self, prototype_id, target_prototype_id=None):
        """
        :param prototype_id: ID of prototype to remove
        :param target_prototype_id: When the 2 prototypes are almost identical, you don't just want to delete
        the prototype, but also transfer its weight to the other prototype. target_prototype_id is used to specify
        the location of the prototype where the weight from prototype_id will be transferred. The result of softmax
        should remain unchanged (as long as the prototypes represented the classes in the same way). If the prototype
        is completely irrelevant (e.g. its weight in the FC layer is 0), simply remove it and let this parameter be None
        """
        with torch.no_grad():
            if target_prototype_id is not None:
                self.fc1.weight[target_prototype_id] += self.fc1.weight[prototype_id]

            self.prototypes.prototypes = torch.nn.Parameter(torch.cat([
                self.prototypes.prototypes[0:prototype_id],
                self.prototypes.prototypes[prototype_id + 1:]
            ]))
            self.prototypes.ones = torch.nn.Parameter(torch.cat([
                self.prototypes.ones[0:prototype_id],
                self.prototypes.ones[prototype_id + 1:]
            ]))
            self.fc1.weight = torch.nn.Parameter(torch.cat([
                self.fc1.weight[:, 0:prototype_id],
                self.fc1.weight[:, prototype_id + 1:]
            ], dim=1))
        print(f'Prototype {prototype_id} was removed')

    def _add_prototype(self):
        """Add prototype add last position
        https://discuss.pytorch.org/t/how-can-i-insert-a-row-into-a-floattensor/18049
        """
        with torch.no_grad():
            prototype_size = self.prototypes.prototypes.shape[1]
            self.prototypes.prototypes = torch.nn.Parameter(torch.cat([
                self.prototypes.prototypes,
                torch.rand([1, prototype_size, 1]).to(self.device)
            ]))
            self.prototypes.ones = torch.nn.Parameter(torch.cat([
                self.prototypes.ones,
                torch.ones([1, prototype_size, 1]).to(self.device)
            ]))

            fan_out_fc, fan_in_fc = self.fc1.weight.shape
            new_weight = torch.nn.Parameter(torch.rand([1, 1]), requires_grad=True)
            std = math.sqrt(2.0 / float(fan_out_fc + fan_in_fc + 1))
            _no_grad_normal_(new_weight, 0., std)
            self.fc1.weight = torch.nn.Parameter(torch.cat([self.fc1.weight, new_weight.to(self.device)], dim=1))
            print(f'Added new prototype, new number of prototypes: {self.prototypes.prototypes.shape[0]}')

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate, eps=1e-8, weight_decay=0.1)
        return {
            'optimizer': optimizer,
            'lr_scheduler': ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1),
            'monitor': f'val_loss_{self.fold_id}'
        }

    @classmethod
    def from_params_and_dataset(cls, train_df, valid_df, params, fold_id):
        TEXT, LABEL, train_loader, val_loader = get_dataset(train_df, valid_df, params.batch_size, params.cache, gpus=1)
        model = cls(vocab_size=len(TEXT.vocab), embedding_dim=TEXT.vocab.vectors.shape[1], fold_id=fold_id,
                    vocab_itos=TEXT.vocab.itos, **params)
        model.embedding.weight.data.copy_(TEXT.vocab.vectors)
        return model, train_loader, val_loader
