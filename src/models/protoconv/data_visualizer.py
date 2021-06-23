from collections import namedtuple, defaultdict
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from utils import html_escape


class DataVisualizer:
    def __init__(self, model):
        self.model = model
        self.separator = '=' * 45

        self.model.eval()
        self.model.cuda()

        self.fc_weights = self.local(self.model.fc1.weight).squeeze(0)
        self.prototypes_words: List[List[str]] = []
        self.prototypes: List[str] = []
        self.enabled_mask = self.local(self.model.enabled_prototypes_mask)
        self.context = self.model.conv_filter_size // 2
        self.padding = self.model.conv_padding
        self.proto_len = self.model.conv_filter_size

        self.tokens_to_phrases()

    def tokens_to_phrases(self):
        words_matrix = self.model.prototype_tokens.tolist()
        for tokens_list in words_matrix:
            words = [self.model.itos[int(token)] for token in tokens_list]
            words = [w for w in words if w not in ['<START>', '<END>', '<pad>']]
            self.prototypes_words.append(words)
            self.prototypes.append(' '.join(words))

    @torch.no_grad()
    def visualize_prototypes(self, output_file_path=None):
        lines = []
        used_ids = self.used_prototypes_ids()
        sorted_by_weights = sorted(used_ids, key=lambda x: self.fc_weights[x], reverse=True)
        for relative_id, prototype_id in enumerate(sorted_by_weights):
            lines.append(f'<b>Prototype {relative_id + 1}</b> (weight {self.fc_weights[prototype_id]:.3f}): '
                         f'{html_escape(self.prototypes[prototype_id])}')
        text = '<br>'.join(lines)

        if output_file_path is not None:
            with open(output_file_path, 'w') as f:
                f.write(text)

        return text

    @torch.no_grad()
    def predict(self, tokens, true_label=None, output_file_path=None, top3=True):
        from models.protoconv.return_wrappers import PrototypeDetailPrediction
        output: PrototypeDetailPrediction = self.model(tokens)
        similarities = self.local(self.model.dist_to_sim['log'](output.min_distances.squeeze(0)))
        evidence = (similarities * self.fc_weights)
        sorting_indexes = np.argsort(evidence)
        enabled_prototypes_indexes = [i for i in sorting_indexes if self.model.enabled_prototypes_mask[i]]
        positive_protos_idxs = [[1, i] for i in enabled_prototypes_indexes[::-1] if evidence[i] > 0]
        negative_protos_idxs = [[0, i] for i in enabled_prototypes_indexes if evidence[i] < 0]

        sum_of_evidence = {
            0: np.sum(evidence[self.fc_weights < 0]) * -1,
            1: np.sum(evidence[self.fc_weights > 0])
        }

        y_pred: int = int(output.logits > 0)

        words = [self.model.itos[j] for j in list(tokens[0]) if self.model.itos[j] not in ['<START>', '<END>']]
        text = html_escape(" ".join(words))

        VisRepresentation = namedtuple("VisRepresentation", "patch_text proto_text similarity weight evidence")
        prototypes_vis_per_class = defaultdict(list)

        if top3:
            negative_protos_idxs = negative_protos_idxs[:3]
            positive_protos_idxs = positive_protos_idxs[:3]

        for class_id, prototype_idx in negative_protos_idxs + positive_protos_idxs:
            patch_center_id = np.argmin(self.local(output.distances)[0, prototype_idx, :])
            if len(words) > self.context:
                patch_center_id = min(max(self.context + 1, patch_center_id), len(words) - 1 - self.context - 1)

            first_index = max(0, patch_center_id - self.context)
            last_index = min(patch_center_id + self.context + 1, len(words) - 1)

            patch_words = words[first_index:last_index]
            patch_words_str = html_escape(' '.join(w for w in patch_words if w not in ['<START>', '<END>']))
            prototype_html = html_escape(self.prototypes[prototype_idx])
            multiplier = 1 if class_id else -1
            prototypes_vis_per_class[class_id].append(
                VisRepresentation(patch_words_str, prototype_html, similarities[prototype_idx],
                                  self.fc_weights[prototype_idx] * multiplier, evidence[prototype_idx] * multiplier)
            )

        class_desc = {0: 'Negative', 1: 'Positive'}

        y_true_text = ''
        if true_label is not None:
            y_true_text = f', <b>Gold standard</b>: {class_desc[true_label]}'

        lines = [f'<b>Input example</b>: {text} <br><br>'
                 f'<b>Prediction</b>: {class_desc[y_pred]}{y_true_text}<br>']

        for class_id, representations in prototypes_vis_per_class.items():
            lines.append(f'Evidence for {class_desc[class_id]} sentiment:')
            lines.append('<table style="width:800px"><tr><td><b>Prototype</b></td><td><b>Most similar phrase</b></td>'
                         '<td><b>Similarity * Weight</b></td></tr>')
            for repr in representations:
                line = f'<tr><td><span">{repr.proto_text} </span> </td> <td> {repr.patch_text} </td> <td>{repr.similarity:.2f} * {repr.weight:.2f} = <b>{repr.evidence:.2f}</b></td></tr>'
                lines[-1] += line
            lines[-1] += '</table>'
            lines[-1] += f'Sum of evidence: <b>{sum_of_evidence[class_id]:.2f}</b><br>'

        text = '<br>'.join(lines)

        if output_file_path is not None:
            with open(output_file_path, 'w') as f:
                f.write(text)

        return text

    @torch.no_grad()
    def visualize_similarity(self):
        p = self.model.prototypes.prototypes.detach().cpu().view(1, self.model.prototypes.prototypes.shape[0], -1)
        distances_matrix = torch.cdist(p, p, p=2).squeeze(0)
        enabled_mask = self.model.enabled_prototypes_mask.bool()

        dist_mat_enabled_proto = distances_matrix[enabled_mask][:, enabled_mask]
        labels = list(np.array(self.prototypes)[enabled_mask.cpu()])

        plt.subplots(figsize=(20, 10))
        ax = sns.heatmap(dist_mat_enabled_proto, xticklabels=labels, yticklabels=labels)
        plt.xticks(rotation=30)

        return ax

    @torch.no_grad()
    def visualize_random_predictions(self, dataloader, n=5, output_file_path=None):
        dataloader.batch_size = 1
        indexes = np.random.choice(len(dataloader.dataset), n, replace=False)
        lines = []

        for i, batch in enumerate(dataloader):
            if i in indexes:
                lines.append(self.predict(batch.text, true_label=batch.label.int().tolist()[0]))
                lines.append(self.separator)

        text = '<br>'.join(lines)

        if output_file_path is not None:
            with open(output_file_path, 'w') as f:
                f.write(text)

        return text

    def used_prototypes_ids(self):
        return list(np.where(self.enabled_mask > 0)[0])

    @staticmethod
    def local(x):
        return x.detach().cpu().numpy()
