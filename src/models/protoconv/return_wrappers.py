from collections import namedtuple

LossesWrapper = namedtuple('LossesWrapper', 'loss cross_entropy clustering_loss separation_loss l1 accuracy l1_words')


class PrototypeDetailPrediction:
    def __init__(self, latent_space, distances, logits, min_distances):
        self.latent_space = latent_space
        self.distances = distances
        self.logits = logits
        self.min_distances = min_distances
