from collections import namedtuple

LossesWrapper = namedtuple('LossesWrapper', 'loss cross_entropy clustering_loss separation_loss l1 accuracy')


class PrototypeDetailPrediction:
    def __init__(self, latent_space, distances, logits, min_distances, tokens_per_kernel=None):
        self.latent_space = latent_space
        self.distances = distances
        self.logits = logits
        self.min_distances = min_distances
        self.tokens_per_kernel = tokens_per_kernel
