from collections import namedtuple

PrototypeDetailPrediction = namedtuple('PrototypeDetailPrediction', 'latent_space distances logits min_distances')
LossesWrapper = namedtuple('LossesWrapper', 'loss cross_entropy clustering_loss separation_loss l1 accuracy')
