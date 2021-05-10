from collections import namedtuple

import numpy as np

from models.protoconv.data_visualizer import html_escape

LossesWrapper = namedtuple('LossesWrapper', 'loss cross_entropy clustering_loss separation_loss l1 accuracy')


class PrototypeDetailPrediction:
    def __init__(self, latent_space, distances, logits, min_distances):
        self.latent_space = latent_space
        self.distances = distances
        self.logits = logits
        self.min_distances = min_distances

class PrototypeRepresentation:
    def __init__(self, best_distance, distances, tokens, words, prototype_weight, enabled=1, absolute_id=None):
        self.best_patch_distance = best_distance
        self.patch_distances = distances
        self.tokens = tokens
        self.words = words
        self.prototype_weight = prototype_weight
        self.enabled = enabled
        self.absolute_id = absolute_id

    def __lt__(self, other):
        return self.best_patch_distance > other.best_patch_distance  # max-heap

    def __repr__(self):
        return f'PR(dist:{self.best_patch_distance})'

    def to_html_heatmap(self, weights: list):
        assert len(weights) == len(self.words), 'Every word has to have a weight'
        highlighted_example = []
        for word, scaled_sim in zip(self.words, weights):
            highlighted_example.append(
                f'<span style="background-color:rgba(135,206,250,{str(scaled_sim)});"> '
                f'{html_escape(word)} </span>')
        highlighted_example = ' '.join(highlighted_example)
        return highlighted_example

    def to_html_bolded(self, context=3):
        best_dist_id = np.argmin(self.patch_distances)
        highlighted_example = []
        for i, word in enumerate(self.words):
            font_weight = 'normal'
            if i - context <= best_dist_id <= i + context:
                font_weight = 'bold'
            highlighted_example.append(
                f'<span style="font-weight:{font_weight};">{html_escape(word)} </span>')
        highlighted_example = ' '.join(highlighted_example)
        return highlighted_example

    def to_html_short(self, context):
        best_dist_id = np.argmin(self.patch_distances)
        highlighted_example = []
        for i, word in enumerate(self.words):
            if i - context <= best_dist_id <= i + context:
                highlighted_example.append(f'<span>{html_escape(word)} </span>')
        highlighted_example = ' '.join(highlighted_example)
        return highlighted_example