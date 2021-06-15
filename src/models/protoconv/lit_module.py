import math

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import BCEWithLogitsLoss
from torch.nn.init import calculate_gain
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from embeddings_dataset_utils import get_dataset
from models.conv_block import ConvolutionalBlock
from models.protoconv.prototype_layer import PrototypeLayer
from models.protoconv.return_wrappers import LossesWrapper, PrototypeDetailPrediction


class ProtoConvLitModule(pl.LightningModule):
    dist_to_sim = {
        'linear': lambda x: -x,
        'log': lambda x: torch.log((x + 1) / (x + 1e-4))
    }

    def __init__(self, vocab_size, embedding_dim, fold_id=1, lr=1e-3, static_embedding=True,
                 pc_project_prototypes_every_n=4, pc_sim_func='log', pc_separation_threshold=10,
                 pc_number_of_prototypes=16, pc_conv_filters=32, pc_ce_loss_weight=1, pc_sep_loss_weight=0,
                 pc_cls_loss_weight=0, pc_l1_loss_weight=0, pc_conv_filter_size=3, pc_prototypes_init='rand', itos=None,
                 pc_dynamic_number=True, *args, **kwargs):
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
        self.conv_filter_size = pc_conv_filter_size
        self.conv_padding = self.conv_filter_size // 2
        self.conv_padding_mode = 'zeros'

        self.prototypes_init = pc_prototypes_init
        self.itos = itos

        self.prototype_similarity_threshold = 0.2
        self.prototype_importance_threshold = 0.002

        self.increment_number_of_prototypes = 2
        self.first_trim_after_projection_epoch = -1  # count from 0

        self.max_number_of_prototypes = 400
        self.dynamic_number = pc_dynamic_number

        self.current_prototypes_number = self.number_of_prototypes
        self.enabled_prototypes_mask = nn.Parameter(torch.cat([
            torch.ones(self.current_prototypes_number),
            torch.zeros(self.max_number_of_prototypes - self.current_prototypes_number)
        ]), requires_grad=False)

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1 = ConvolutionalBlock(self.embedding_dim, self.conv_filters, kernel_size=self.conv_filter_size,
                                        padding=self.conv_padding, stride=1,
                                        padding_mode=self.conv_padding_mode)
        self.prototypes = PrototypeLayer(channels_in=self.conv_filters,
                                         number_of_prototypes=self.max_number_of_prototypes,
                                         initialization=self.prototypes_init)
        self.fc1 = nn.Linear(self.max_number_of_prototypes, 1, bias=False)

        self.prototype_tokens = nn.Parameter(torch.zeros([self.max_number_of_prototypes, self.conv_filter_size],
                                                         dtype=torch.int), requires_grad=False)

        if static_embedding:
            self.embedding.weight.requires_grad = False

        self._init_fc_layer()
        self._zeroing_disabled_prototypes()

        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        self.loss = BCEWithLogitsLoss()

    def forward(self, x):
        embedding = self.embedding(x).permute((0, 2, 1))
        latent_space = self.conv1(embedding)

        padded_tokens = F.pad(x, (self.conv_padding, self.conv_padding), 'constant')
        tokens_per_kernel = padded_tokens.unfold(1, self.conv_filter_size, 1)

        distances = self.prototypes(latent_space)
        min_dist = self._min_pooling(distances)
        similarity = self.dist_to_sim[self.sim_func](min_dist)
        masked_similarity = similarity * self.enabled_prototypes_mask
        logits = self.fc1(masked_similarity).squeeze(1)
        return PrototypeDetailPrediction(latent_space, distances, logits, min_dist, tokens_per_kernel)

    @torch.no_grad()
    def on_train_epoch_start(self, *args, **kwargs):
        if self.dynamic_number is True and self.current_epoch >= (self.first_trim_after_projection_epoch + 1):
            print("Dynamic curr_epoch>=firstrim+1")
            self._add_prototypes(self.increment_number_of_prototypes)

    def training_step(self, batch, batch_nb):
        losses = self.learning_step(batch, self.train_acc)
        self.log_all_metrics('train', losses)
        return {'loss': losses.loss}

    @torch.no_grad()
    def on_validation_epoch_start(self, *args, **kwargs):
        if self._is_projection_prototype_epoch():
            self._project_prototypes()

    @torch.no_grad()
    def _project_prototypes(self):
        self.eval()
        projected_prototypes = torch.zeros_like(self.prototypes.prototypes, device=self.device)
        min_distances_prototype_example = torch.full([self.max_number_of_prototypes], float('inf'), device=self.device)
        prototype_tokens = torch.zeros([self.max_number_of_prototypes, self.conv_filter_size], dtype=torch.int,
                                       device=self.device)
        for batch in self.train_dataloader():
            predictions = self(batch.text)
            for pred_id in range(len(predictions.logits)):
                tokens_per_kernel = predictions.tokens_per_kernel[pred_id].squeeze(0)
                latent_space = predictions.latent_space[pred_id].squeeze(0)
                distances = predictions.distances[pred_id].squeeze(0)

                assert latent_space.shape[1] == tokens_per_kernel.shape[0] == distances.shape[1]

                best_distances, best_distances_idx = torch.min(distances, dim=1)
                best_latents_to_prototype = latent_space.permute(1, 0)[best_distances_idx].unsqueeze(2)
                best_tokens_to_prototype = tokens_per_kernel[best_distances_idx]

                update_mask = best_distances < min_distances_prototype_example
                update_indexes = torch.where(update_mask == 1)

                min_distances_prototype_example[update_indexes] = best_distances[update_indexes]
                projected_prototypes[update_indexes] = best_latents_to_prototype[update_indexes]
                prototype_tokens[update_indexes] = best_tokens_to_prototype[update_indexes].int()

        self.prototypes.prototypes.data.copy_(torch.tensor(projected_prototypes))
        self.prototype_tokens.data.copy_(torch.tensor(prototype_tokens))

        if self.dynamic_number is True and self.current_epoch >= self.first_trim_after_projection_epoch:
            print('Dynamic Currepocj>=firsttrim')
            self._remove_non_important_prototypes()
            self._merge_similar_prototypes()

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
        self.log(f'number_of_prototypes_{self.fold_id}', self.current_prototypes_number, prog_bar=True, on_step=False,
                 on_epoch=True)

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
        prot = prototypes.view(1, prototypes.shape[0], -1)  # [1, prototypes, all_params_of_prototype]
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

    @torch.no_grad()
    def _remove_non_important_prototypes(self):
        non_important_prototypes_idxs = (abs(self.fc1.weight[0]) <= self.prototype_importance_threshold) \
                                        * self.enabled_prototypes_mask
        remove_ids = torch.nonzero(non_important_prototypes_idxs, as_tuple=False).squeeze(1).tolist()
        if 1 <= len(remove_ids):
            shorten_ids = remove_ids[:self.current_prototypes_number - 4]
            self._remove_prototypes(shorten_ids)
            print(f'Prototypes {remove_ids}, were removed')

    @torch.no_grad()
    def _merge_similar_prototypes(self):
        used_prototypes = torch.where(self.enabled_prototypes_mask.data == 1)[0].tolist()
        prot = self.prototypes.prototypes.view(1, self.prototypes.prototypes.shape[0], -1)
        distances_matrix = torch.cdist(prot, prot, p=2).squeeze(0)  # [prototypes, prototypes]
        argsorted = torch.argsort(distances_matrix, dim=1)

        from_list, to_list = [], []
        for prototype_idx in range(len(argsorted)):
            if prototype_idx not in used_prototypes or prototype_idx in to_list + from_list:
                continue
            for target_proto_idx in range(len(argsorted[0])):
                if target_proto_idx not in used_prototypes or prototype_idx == target_proto_idx or \
                        target_proto_idx in to_list + from_list:
                    continue
                elif abs(distances_matrix[prototype_idx, target_proto_idx]) <= self.prototype_similarity_threshold:
                    from_list.append(prototype_idx)
                    to_list.append(target_proto_idx)
                    break

        if 1 <= len(to_list) <= self.current_prototypes_number - 2:
            self._remove_prototypes(to_list, from_list)
            print(f'Prototypes {to_list}, {from_list} were merged')

    def _add_prototypes(self, quantity):
        added_idx = [self._add_prototype() for _ in range(quantity)]
        print(f'Added: {added_idx} prototypes')

    @torch.no_grad()
    def _remove_prototypes(self, prototype_ids: list, target_prototype_ids=None):
        """
        :param prototype_ids: list IDs of prototypes to remove
        :param target_prototype_ids: When the 2 prototypes are almost identical, you don't just want to delete
        the prototype, but also transfer its weight to the other prototype. target_prototype_id is used to specify
        the location of the prototype where the weight from prototype_id will be transferred. The result of softmax
        should remain unchanged (as long as the prototypes represented the classes in the same way). If the prototype
        is completely irrelevant (e.g. its weight in the FC layer is 0), simply remove it and let this parameter be None
        """
        zero_indices = torch.nonzero(self.enabled_prototypes_mask == 0, as_tuple=False).squeeze(1).tolist()

        assert type(prototype_ids) == list
        assert len(set(zero_indices) & set(prototype_ids)) == 0, \
            f'You are trying to remove prototypes that dont exist: {set(zero_indices) & set(prototype_ids)}'

        if target_prototype_ids is not None:
            assert type(target_prototype_ids) == list
            assert len(set(prototype_ids) & set(target_prototype_ids)) == 0
            assert len(prototype_ids) == len(target_prototype_ids)
            assert len(set(zero_indices) & set(target_prototype_ids)) == 0, \
                f'You are trying to add value to prototypes that dont exist: {set(zero_indices) & set(prototype_ids)}'
            self.fc1.weight.data[0, target_prototype_ids] += self.fc1.weight.data[0, prototype_ids]

        self.current_prototypes_number -= len(prototype_ids)
        self.enabled_prototypes_mask[prototype_ids] = 0
        self._zeroing_disabled_prototypes()

    @torch.no_grad()
    def _add_prototype(self):
        if self.current_prototypes_number == self.max_number_of_prototypes:
            return None
        zero_indices = torch.nonzero(self.enabled_prototypes_mask == 0, as_tuple=False).squeeze(1)
        new_prototype_id = zero_indices[0].item()

        std = calculate_gain('leaky_relu', math.sqrt(5)) / math.sqrt(self.current_prototypes_number + 1)
        bound = math.sqrt(3.0) * std
        self.fc1.weight.data[0, new_prototype_id].uniform_(-bound, bound)
        self.prototypes.prototypes.data[new_prototype_id].uniform_(0, 1)
        self.enabled_prototypes_mask[new_prototype_id] = 1
        self.current_prototypes_number += 1
        return new_prototype_id

    @torch.no_grad()
    def _init_fc_layer(self):
        std = calculate_gain('leaky_relu', math.sqrt(5)) / math.sqrt(self.current_prototypes_number)
        bound = math.sqrt(3.0) * std
        self.fc1.weight.uniform_(-bound, bound)  # kaiming_uniform_

    @torch.no_grad()
    def _zeroing_disabled_prototypes(self):
        self.prototypes.prototypes.data[~self.enabled_prototypes_mask.bool()] *= 0
        self.fc1.weight.data[:, ~self.enabled_prototypes_mask.bool()] *= 0

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate, eps=1e-8, weight_decay=0.1)
        lr_scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', patience=4, factor=0.1, verbose=True,
                                           threshold=0.005),
            'name': f'learning_rate_{self.fold_id}',
            'monitor': f'val_loss_{self.fold_id}'
        }
        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
        }

    @classmethod
    def from_params_and_dataset(cls, train_df, valid_df, params, fold_id, embeddings=None):
        TEXT, LABEL, train_loader, val_loader = get_dataset(train_df, valid_df, params.batch_size, gpus=params.gpu,
                                                            vectors=embeddings)
        model = cls(vocab_size=len(TEXT.vocab), embedding_dim=TEXT.vocab.vectors.shape[1], fold_id=fold_id,
                    itos=TEXT.vocab.itos, **params)
        model.embedding.weight.data.copy_(TEXT.vocab.vectors)
        utils = {
            'TEXT': TEXT
        }
        return model, train_loader, val_loader, utils
