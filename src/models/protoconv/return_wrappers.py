from collections import namedtuple

PrototypeDetailPrediction = namedtuple('PrototypeDetailPrediction', 'latent_space distances logits min_distances')
LossesWrapper = namedtuple('LossesWrapper', 'loss cross_entropy clustering_loss separation_loss l1 accuracy')


class PrototypeRepresentation:
    def __init__(self, best_distance, distances, X):
        self.best_patch_distance = best_distance
        self.patch_distances = distances
        self.X = X

    def __lt__(self, other):
        return self.best_patch_distance > other.best_patch_distance  # max-heap

    def __repr__(self):
        return f'PE(dist:{self.best_patch_distance})'
