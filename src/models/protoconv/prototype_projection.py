import torch

from models.protoconv.return_wrappers import PrototypeDetailPrediction


# TODO Better comments
class PrototypeProjection:
    """
    Class responsible for holding and updating state during projection prototype epoch
    """

    def __init__(self, prototype_shape, prototype_words):
        """
        :param prototype_shape: Shape of prototype layer
        :param prototype_words: Number of words in prototype
        """
        self.number_of_prototypes = prototype_shape[0]
        self.prototype_shape = prototype_shape[1:]
        self.prototype_words = prototype_words

    def update(self, predictions: PrototypeDetailPrediction):
        for pred_id in range(len(predictions.logits)):
            tokens_per_kernel = predictions.tokens_per_kernel[pred_id].squeeze(0)
            latent_space = predictions.latent_space[pred_id].squeeze(0)
            distances = predictions.distances[pred_id].squeeze(0)

            best_distances, best_distances_idx = torch.min(distances, dim=1)
            best_latents_to_prototype = latent_space.permute(1, 0)[best_distances_idx].unsqueeze(2)
            best_tokens_to_prototype = tokens_per_kernel[best_distances_idx]

            update_mask = best_distances < self._min_distances_prototype_example
            update_indexes = torch.where(update_mask == 1)

            self._min_distances_prototype_example[update_indexes] = best_distances[update_indexes]
            self._projected_prototypes[update_indexes] = best_latents_to_prototype[update_indexes]
            self._prototype_tokens[update_indexes] = best_tokens_to_prototype[update_indexes].int()

    def get_weights(self):
        return torch.tensor(self._projected_prototypes)

    def get_tokens(self):
        return torch.tensor(self._prototype_tokens)

    def reset(self, device):
        self._projected_prototypes = torch.zeros([self.number_of_prototypes, *self.prototype_shape], device=device)
        self._min_distances_prototype_example = torch.full([self.number_of_prototypes], float('inf'), device=device)
        self._prototype_tokens = torch.zeros([self.number_of_prototypes, self.prototype_words], device=device,
                                             dtype=torch.int)
