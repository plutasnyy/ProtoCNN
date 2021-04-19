import torch

from models.protoconv.return_wrappers import PrototypeDetailPrediction


class PrototypeProjection:
    """
    Class responsible for holding and updating state during projection prototype epoch
    """

    def __init__(self, prototype_shape):
        self.prototype_shape = prototype_shape
        self.number_of_prototypes = prototype_shape[0]

    def update(self, predictions: PrototypeDetailPrediction):
        """
        """
        # assert predictions.distances.shape[1] == predictions.latent_space.shape[1]

        for pred_id in range(len(predictions.logits)):
            latent_space = predictions.latent_space[pred_id].squeeze(0)  # [32,115] [ latent_size, length]
            distances = predictions.distances[pred_id].squeeze(0)  # [16,115]  [prototypes, distances]

            best_distances, best_distances_idx = torch.min(distances, dim=1)
            best_latents_to_prototype = latent_space.permute(1, 0)[best_distances_idx].unsqueeze(2)
            # permute to [115, 32] for correct selecting representations, after selection [prototypes, 32, 1]

            update_mask = best_distances < self._min_distances_prototype_example
            update_indexes = torch.where(update_mask == 1)

            self._min_distances_prototype_example[update_indexes] = best_distances[update_indexes]
            self._projected_prototypes[update_indexes] = best_latents_to_prototype[update_indexes]  # [16,32,1]

    def get_weights(self):
        return torch.tensor(self._projected_prototypes)

    def reset(self, device, number_of_prototypes=None):
        if number_of_prototypes is not None:
            self.number_of_prototypes = number_of_prototypes

        self._projected_prototypes = torch.zeros(self.prototype_shape, device=device)
        self._min_distances_prototype_example = torch.full([self.number_of_prototypes], float('inf'), device=device)
