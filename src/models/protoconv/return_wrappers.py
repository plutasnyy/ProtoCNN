from collections import namedtuple

PrototypeDetailPrediction = namedtuple('PrototypeDetailPrediction', 'latent_space distances logits min_distances')
LossesWrapper = namedtuple('LossesWrapper', 'loss cross_entropy clustering_loss separation_loss l1 accuracy')


class PrototypeRepresentation:
    def __init__(self, best_distance, distances, X):
        self.best_distance = best_distance
        self.distances = distances
        self.similarity = -best_distance
        self.X = X

    def __lt__(self, other):
        return self.similarity < other.similarity  # max-heap

    def __repr__(self):
        return f'PE(dist:{self.best_distance})'
