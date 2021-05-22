import torch
import torch.nn.functional as F

from models.protoconv.return_wrappers import PrototypeDetailPrediction


# TODO Better comments
# TODO Remember prototypes during learning
class PrototypeProjection:
    """
    Class responsible for holding and updating state during projection prototype epoch
    """

    def __init__(self, prototype_shape):
        """
        :param prototype_shape: Shape of prototype layer
        """
        self.number_of_prototypes = prototype_shape[0]
        self.prototype_shape = prototype_shape[1:]
        self.prototype_length = prototype_shape[-1]
        self.padding = self.prototype_length // 2

    def update(self, predictions: PrototypeDetailPrediction, filters_words_importance):
        for pred_id in range(len(predictions.logits)):
            x = predictions.latent_space[pred_id]
            if self.padding >= 1:
                x = F.pad(x, (self.padding, self.padding), 'constant')
            latent_space = x.unfold(dimension=1, step=1, size=self.prototype_length).squeeze(0).permute(1, 0, 2)
            distances = predictions.distances[pred_id].squeeze(0)

            best_distances, best_distances_idx = torch.min(distances, dim=1)
            best_latents_to_prototype = latent_space[best_distances_idx]

            update_mask = best_distances < self._min_distances_prototype_example
            update_indexes = torch.where(update_mask == 1)

            # self._prototype_words_importance[update_indexes] = filters_words_importance[update_indexes]
            self._min_distances_prototype_example[update_indexes] = best_distances[update_indexes]
            self._projected_prototypes[update_indexes] = best_latents_to_prototype[update_indexes]  # [16,32,1]

    def get_weights(self):
        return torch.tensor(self._projected_prototypes)

    def reset(self, device):
        self._projected_prototypes = torch.zeros([self.number_of_prototypes, *self.prototype_shape], device=device)
        self._min_distances_prototype_example = torch.full([self.number_of_prototypes], float('inf'), device=device)
        # self._prototype_words_importance = torch.zeros([self.number_of_prototypes, self.prototype_length],
        #                                                requires_grad=False, device=device)
