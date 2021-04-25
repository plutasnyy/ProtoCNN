import heapq
import html
from ast import literal_eval
from configparser import ConfigParser
from io import BytesIO

import click
import comet_ml
import cv2
import pandas as pd
import torch
from easydict import EasyDict

from models.embeddings_dataset_utils import get_dataset
from models.protoconv.return_wrappers import PrototypeRepresentation


def html_escape(text):
    return html.escape(text)


def local(x):
    return x.detach().cpu().numpy()


@click.command()
@click.option('--experiment', required=True, type=str, help='For example 90b50f0dc0e54b9e89ffab66db34db10')
@click.option('--weights-path', required=True, type=str, help='For example epoch=25-val_mae=8.2030.ckpt')
@click.option('--fold', required=True, type=int)
@click.option('-k', required=True, type=int, help='Number of most similar examples to prototype')
def visualize(experiment, weights_path, fold, k):
    config = ConfigParser()
    config.read('config.ini')

    comet_config = EasyDict(config['cometml'])
    comet_api = comet_ml.api.API(api_key=comet_config.apikey)
    experiment = comet_api.get(project_name=comet_config.projectname, workspace=comet_config.workspace,
                               experiment=experiment)

    dataset = experiment.get_parameters_summary('data_set')['valueCurrent']
    kfold_split_id = list(
        filter(lambda x: x['fileName'] == 'kfold_split_indices.csv', experiment.get_asset_list())
    )[0]['assetId']
    kfold_split_binary = experiment.get_asset(kfold_split_id, return_type="binary")
    kfold_split = pd.read_csv(BytesIO(kfold_split_binary)).iloc[fold]

    train_index = literal_eval(kfold_split['train_indices'])
    val_index = literal_eval(kfold_split['val_indices'])
    test_index = literal_eval(kfold_split['test_indices'])

    df_dataset = pd.read_csv(f'data/{dataset}/data.csv')
    train_df, valid_df = df_dataset.iloc[train_index + val_index], df_dataset.iloc[test_index]

    TEXT, LABEL, train_loader, val_loader = get_dataset(train_df, valid_df, batch_size=1, cache=None)

    from models.protoconv.lit_module import ProtoConvLitModule
    model = ProtoConvLitModule.load_from_checkpoint('checkpoints/' + weights_path)
    model.vocab_itos = TEXT.vocab.itos
    visualize_model(model, train_loader, k, 'prototype_visualization.html')


def visualize_model(model, data_loader, k_most_similar, file_name):
    model.eval()
    model.cuda()

    data_loader.batch_size = 1
    vocab_int_to_string = model.vocab_itos
    n_prototypes = model.prototypes.prototypes.shape[0]

    heaps = [[] for _ in range(n_prototypes)]

    from models.protoconv.return_wrappers import PrototypeDetailPrediction
    with torch.no_grad():
        for X, batch_id in data_loader:
            outputs: PrototypeDetailPrediction = model(X)

            for i in range(n_prototypes):
                if model.enabled_prototypes_mask[i] == 1:
                    prototype_repr = PrototypeRepresentation(
                        float(local(outputs.min_distances.squeeze(0)[i])),
                        local(outputs.distances.squeeze(0)[i]),
                        local(X.squeeze(0))
                    )

                    if len(heaps[i]) < k_most_similar:
                        heapq.heappush(heaps[i], prototype_repr)
                    else:
                        heapq.heappushpop(heaps[i], prototype_repr)

    lines = []
    separator = '-' * 15
    fc_weights = local(model.fc1.weight.squeeze(0))

    for prototype_id in range(n_prototypes):

        if not model.enabled_prototypes_mask[prototype_id]:
            continue

        prototype_weight = round(float(fc_weights[prototype_id]), 4)
        lines.append(f'{separator} <b>Prototype {prototype_id + 1}</b>, weight {prototype_weight} {separator}')

        for example_id, example in enumerate(heapq.nlargest(3, heaps[prototype_id])):
            best_sim = model.dist_to_sim['log'](torch.tensor(example.best_patch_distance)).item()

            best_dist_round = round(float(example.best_patch_distance), 4)
            best_sim_round = round(float(best_sim), 4)
            lines.append(f'Example {example_id + 1}, best distance {best_dist_round} (similarity {best_sim_round})')

            words = [vocab_int_to_string[j] for j in list(example.X)]

            similarities = model.dist_to_sim['log'](torch.tensor(example.patch_distances)).numpy()
            similarities_scaled = cv2.resize(similarities, dsize=(1, len(words)), interpolation=cv2.INTER_LINEAR)
            min_d, max_d = min(similarities_scaled), max(similarities_scaled)
            similarity_weight = ((similarities_scaled - min_d) / (max_d - min_d)).squeeze(1)

            highlighted_example = []
            for word, scaled_sim in zip(words, list(similarity_weight)):
                highlighted_example.append(
                    f'<span style="background-color:rgba(135,206,250,{str(scaled_sim)});"> {html_escape(word)} </span>')
            highlighted_example = ' '.join(highlighted_example)

            lines.append(highlighted_example)
            lines.append('')

    text = '<br>'.join(lines)

    with open(file_name, 'w') as f:
        f.write(text)


if __name__ == '__main__':
    visualize()
