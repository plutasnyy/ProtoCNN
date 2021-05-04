import heapq
import html
from typing import List

import cv2
import torch

from models.protoconv.return_wrappers import PrototypeDetailPrediction, PrototypeRepresentation


class DataVisualizer:
    def __init__(self, model, train_loader, val_loader=None, vocab_itos=None):
        self.prototypes: List[List[PrototypeRepresentation]] = []
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.vocab_itos = vocab_itos

        self.model.eval()
        self.model.cuda()

        self.fc_weights = round(float(self.local(self.model.fc1.weight.squeeze(0))), 4)

        self.train_loader.batch_size = 1
        if self.val_loader:
            self.val_loader.batch_size = 1

        self.find_prototypes_representation()

    @torch.no_grad()
    def find_prototypes_representation(self, k_most_similar=3):
        heaps = [[] for _ in range(self.number_of_prototypes)]

        for X, batch_id in self.train_loader:
            outputs: PrototypeDetailPrediction = self.model(X)

            for i in range(self.number_of_prototypes):
                prototype_repr = PrototypeRepresentation(
                    best_distance=self.local(outputs.min_distances.squeeze(0)[i]).float(),
                    distances=self.local(outputs.distances.squeeze(0)[i]),
                    tokens=self.local(X.squeeze(0)),
                    prototype_weight=self.fc_weights[i],
                    enabled=self.model.enabled_prototypes_mask[i],
                    absolute_id=i
                )

                if len(heaps[i]) < k_most_similar:
                    heapq.heappush(heaps[i], prototype_repr)
                else:
                    heapq.heappushpop(heaps[i], prototype_repr)

            for absolute_id, heap in enumerate(heaps):
                self.prototypes[absolute_id] = heapq.nlargest(k_most_similar, heap)

    @torch.no_grad()
    def visualize_prototypes_as_heatmap(self, output_file_path='prototypes.html'):
        lines = []
        separator = '-' * 15

        for relative_id, representations_list in enumerate(self.filter_enabled_prototypes(self.prototypes)):
            lines.append(
                f'{separator} <b>Prototype {relative_id + 1}</b>, '
                f'weight {representations_list[0].prototype_weight} {separator}')

            for example_id, example in enumerate(representations_list):
                best_sim = self.model.dist_to_sim['log'](torch.tensor(example.best_patch_distance)).item()

                best_dist_round = round(float(example.best_patch_distance), 4)
                best_sim_round = round(float(best_sim), 4)
                lines.append(f'Example {example_id + 1}, best distance {best_dist_round} (similarity {best_sim_round})')

                words = [self.vocab_itos[j] for j in list(example.tokens)]
                weights = self.weights_from_distances(distances=example.patch_distances, target_len=len(words))

                highlighted_example = []
                for word, scaled_sim in zip(words, list(weights)):
                    highlighted_example.append(
                        f'<span style="background-color:rgba(135,206,250,{str(scaled_sim)});"> '
                        f'{self.html_escape(word)} </span>')
                highlighted_example = ' '.join(highlighted_example)

                lines.append(highlighted_example)
                lines.append('')

        text = '<br>'.join(lines)

        with open(output_file_path, 'w') as f:
            f.write(text)

    def weights_from_distances(self, distances, target_len):
        similarities = self.model.dist_to_sim['log'](torch.tensor(distances)).numpy()
        similarities_scaled = cv2.resize(similarities, dsize=(1, target_len), interpolation=cv2.INTER_LINEAR)
        min_d, max_d = min(similarities_scaled), max(similarities_scaled)
        similarity_weight = ((similarities_scaled - min_d) / (max_d - min_d)).squeeze(1)
        return similarity_weight

    @property
    def number_of_prototypes(self):
        return self.model.prototypes.prototypes.shape[0]

    @staticmethod
    def filter_enabled_prototypes(prototypes_representations: List[List[PrototypeRepresentation]]):
        return filter(lambda p: p[0].enabled, prototypes_representations)

    @staticmethod
    def local(x):
        return x.detach().cpu().numpy()

    @staticmethod
    def html_escape(text):
        return html.escape(text)
