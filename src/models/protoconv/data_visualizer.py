import heapq
from collections import namedtuple, defaultdict
from operator import itemgetter
from typing import List

import cv2
import numpy as np
import torch
from tqdm.contrib import tenumerate

from utils import html_escape


class DataVisualizer:
    def __init__(self, model, train_loader, vocab_itos=None):
        from models.protoconv.prototype_representation import PrototypeRepresentation
        self.prototypes: List[List[PrototypeRepresentation]] = []
        self.model = model
        self.train_loader = train_loader
        self.vocab_itos = vocab_itos
        self.separator = '-' * 15

        self.model.eval()
        self.model.cuda()

        self.fc_weights = self.local(self.model.fc1.weight).squeeze(0)
        self.context = self.model.conv_filter_size // 2

        self.train_loader.shuffle = False
        self.train_loader.batch_size = 1

        self.find_prototypes_representation()

    @torch.no_grad()
    def find_prototypes_representation(self, k_most_similar=3):
        from models.protoconv.return_wrappers import PrototypeDetailPrediction
        from models.protoconv.prototype_representation import PrototypeRepresentation

        heaps = [[] for _ in range(self.number_of_prototypes)]
        for example_id, (X, y) in tenumerate(self.train_loader, total=len(self.train_loader)):
            outputs: PrototypeDetailPrediction = self.model(X)
            tokens = self.local(X.squeeze(0))
            words = [self.vocab_itos[j] for j in list(tokens)]

            for i in range(self.number_of_prototypes):
                prototype_repr = PrototypeRepresentation(
                    best_distance=self.local(outputs.min_distances.squeeze(0)[i]),
                    distances=self.local(outputs.distances.squeeze(0)[i]),
                    tokens=tokens,
                    words=words,
                    prototype_weight=self.fc_weights[i],
                    enabled=self.model.enabled_prototypes_mask[i],
                    absolute_id=i
                )

                if len(heaps[i]) < k_most_similar:
                    heapq.heappush(heaps[i], prototype_repr)
                else:
                    heapq.heappushpop(heaps[i], prototype_repr)
                # if example_id > 5:
                #     break

        for heap in heaps:
            self.prototypes.append(heapq.nlargest(k_most_similar, heap))

    @torch.no_grad()
    def visualize_prototypes_as_heatmap(self, output_file_path=None):
        lines = []
        for relative_id, representations_list in enumerate(self.filter_enabled_prototypes(self.prototypes)):
            lines.append(
                f'{self.separator} <b>Prototype {relative_id + 1}</b>, '
                f'weight {representations_list[0].prototype_weight:.4f} {self.separator}')

            for example_id, prototype in enumerate(representations_list):
                best_sim = self.model.dist_to_sim['log'](torch.tensor(prototype.best_patch_distance)).item()
                lines.append(f'Example {example_id + 1}, best distance {prototype.best_patch_distance:.4f} '
                             f'(similarity {best_sim:.4f})')
                weights = self.weights_from_distances(distances=prototype.patch_distances,
                                                      target_len=len(prototype.words))
                weights = self.expand_weight_to_the_context(weights, context=self.context)
                lines.append(f'{prototype.to_html_heatmap(weights)}<br>')

        text = '<br>'.join(lines)

        if output_file_path is not None:
            with open(output_file_path, 'w') as f:
                f.write(text)

        return text

    @torch.no_grad()
    def visualize_prototypes_as_bold(self, output_file_path=None):
        lines = []
        for relative_id, prototype in enumerate(self.filter_used(self.prototypes)):
            lines.append(f'{self.separator} <b>Prototype {relative_id + 1}</b> '
                         f'weight {prototype.prototype_weight:.4f} {self.separator} <br>'
                         f'{prototype.to_html_bolded(self.context)}<br>')

        text = '<br>'.join(lines)

        if output_file_path is not None:
            with open(output_file_path, 'w') as f:
                f.write(text)

        return text

    @torch.no_grad()
    def visualize_prototypes_as_short(self, output_file_path=None):
        lines = []
        for relative_id, prototype in enumerate(self.filter_used(self.prototypes)):
            lines.append(f'{self.separator} <b>Prototype {relative_id + 1}</b> '
                         f'weight {prototype.prototype_weight:.4f} {self.separator} <br>'
                         f'{prototype.to_html_short(self.context)}<br>')

        text = '<br>'.join(lines)

        if output_file_path is not None:
            with open(output_file_path, 'w') as f:
                f.write(text)

        return text

    @torch.no_grad()
    def predict(self, tokens, true_label=None, output_file_path=None):
        from models.protoconv.return_wrappers import PrototypeDetailPrediction
        from models.protoconv.prototype_representation import PrototypeRepresentation

        output: PrototypeDetailPrediction = self.model(tokens)
        closest_prototypes: List[PrototypeRepresentation] = list(self.filter_most_similar(self.prototypes))
        similarities = self.local(self.model.dist_to_sim['log'](output.min_distances.squeeze(0)))
        evidence = (similarities * self.fc_weights)
        sorting_indexes = np.argsort(evidence)
        enabled_prototypes_indexes = [i for i in sorting_indexes if self.model.enabled_prototypes_mask[i]]
        positive_protos_idxs = [[1, i] for i in enabled_prototypes_indexes[::-1] if evidence[i] > 0]
        negative_protos_idxs = [[0, i] for i in enabled_prototypes_indexes if evidence[i] < 0]

        sum_of_evidence = {
            0: np.sum(evidence[self.fc_weights < 0]),
            1: np.sum(evidence[self.fc_weights > 0])
        }

        y_pred: int = int(output.logits > 0)

        words = [self.vocab_itos[j] for j in list(tokens[0])]
        text = " ".join(words)

        VisRepresentation = namedtuple("VisRepresentation", "patch_text proto_text similarity weight evidence")
        prototypes_vis_per_class = defaultdict(list)
        for class_id, prototype_idx in negative_protos_idxs+positive_protos_idxs:
            patch_center_id = np.argmin(self.local(output.distances)[0, prototype_idx, :])
            patch_words = words[patch_center_id - self.context:patch_center_id + self.context + 1]
            patch_words_str = html_escape(' '.join(patch_words))
            prototype_html = closest_prototypes[prototype_idx].to_html_short(self.context)
            prototypes_vis_per_class[class_id].append(
                VisRepresentation(patch_words_str, prototype_html, similarities[prototype_idx],
                                  self.fc_weights[prototype_idx], evidence[prototype_idx])
            )

        y_true_text = ''
        if true_label is not None:
            y_true_text = f', <b>True</b>: {true_label}'

        lines = [f'<b>Example</b>: {text} <br><br>'
                 f'<b>Prediction</b>: {y_pred}{y_true_text}<br>']

        for class_id, representations in prototypes_vis_per_class.items():
            lines.append(f'Evidence for class {class_id}:')
            lines.append('<table style="width:100%"><tr><td><b>Input</b></td><td><b>Prototype</b></td>'
                         '<td><b>Similarity * Importance</b></td></tr>')
            for repr in representations:
                line = f'<tr><td><span">{repr.patch_text} </span> </td> ' \
                       f'<td> {repr.proto_text} </td>' + \
                       f'<td>{repr.similarity:.2f} * {repr.weight:.2f} = {repr.evidence:.2f}</td></tr>'
                lines[-1] += line
            lines[-1] += '</table>'
            lines[-1] += f'Sum of evidence for class {class_id}: {sum_of_evidence[class_id]:.2f}<br>'

        text = '<br>'.join(lines)

        if output_file_path is not None:
            with open(output_file_path, 'w') as f:
                f.write(text)

        return text

    def weights_from_distances(self, distances, target_len):
        similarities = self.model.dist_to_sim['log'](torch.tensor(distances)).numpy()
        similarities_scaled = cv2.resize(similarities, dsize=(1, target_len), interpolation=cv2.INTER_LINEAR)
        min_d, max_d = min(similarities_scaled), max(similarities_scaled)
        similarity_weight = ((similarities_scaled - min_d) / (max_d - min_d)).squeeze(1)
        return list(similarity_weight)

    @staticmethod
    def expand_weight_to_the_context(weights, context):
        result_array = np.zeros_like(weights)
        for i, w in enumerate(weights):
            min_range, max_range = max(i - context, 0), min(i + context + 1, len(weights))
            for j in range(min_range, max_range):
                result_array[j] += w
        min_d, max_d = min(result_array), max(result_array)
        result_array = (result_array - min_d) / (max_d - min_d + 1e-4)
        return result_array

    @property
    def number_of_prototypes(self):
        return self.model.prototypes.prototypes.shape[0]

    @staticmethod
    def filter_enabled_prototypes(prototypes_representations):
        return filter(lambda p: p[0].enabled, prototypes_representations)

    @staticmethod
    def local(x):
        return x.detach().cpu().numpy()

    @staticmethod
    def filter_most_similar(prototypes_representations):
        return map(itemgetter(0), prototypes_representations)

    @staticmethod
    def filter_used(prototypes):
        return DataVisualizer.filter_most_similar(DataVisualizer.filter_enabled_prototypes(prototypes))
