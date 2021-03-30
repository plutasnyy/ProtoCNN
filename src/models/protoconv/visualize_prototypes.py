import heapq
import html
from ast import literal_eval
from configparser import ConfigParser
from io import BytesIO

import click
import comet_ml
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from IPython.core.display import display, HTML
from easydict import EasyDict

from models.protoconv.lit_module import ProtoConvLitModule
from models.protoconv.return_wrappers import PrototypeRepresentation
from models.protoconv.train import get_dataset


def html_escape(text):
    return html.escape(text)


@click.command()
@click.option('--experiment', required=True, type=str, help='For example ce132011516346c99185d139fb23c70c')
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

    df_dataset = pd.read_csv(f'data/{dataset}/data.csv')
    train_df, valid_df = df_dataset.iloc[train_index], df_dataset.iloc[val_index]

    TEXT, LABEL, train_loader, val_loader = get_dataset(train_df, valid_df, batch_size=1, cache=None)
    model = ProtoConvLitModule.load_from_checkpoint('checkpoints/' + weights_path, vocab_size=len(TEXT.vocab),
                                                    embedding_dim=TEXT.vocab.vectors.shape[1], lr=1, fold_id=fold)
    model.eval()
    model.cuda()
    n_prototypes = model.prototype.prototypes.shape[0]

    heaps = [[] for _ in range(n_prototypes)]

    with torch.no_grad():
        for X, batch_id in train_loader:
            temp = model.embedding(X).permute((0, 2, 1))
            temp = model.conv1(temp)
            temp = model.prototype(temp)
            prototypes_distances = temp.squeeze(0).detach().cpu().numpy()  # [Prototype, len(X)-4]
            temp = -F.max_pool1d(-temp, temp.size(2))
            best_distances = temp.view(temp.size(0), -1).squeeze(0).detach().cpu().numpy()  # [Prototype]

            for i, (best_dist, dists_in_prot) in enumerate(zip(best_distances, prototypes_distances)):
                prototype_repr = PrototypeRepresentation(best_dist, dists_in_prot, X[0].detach().cpu().numpy())
                if len(heaps[i]) < k:
                    heapq.heappush(heaps[i], prototype_repr)
                else:
                    heapq.heappushpop(heaps[i], prototype_repr)

    result = ''
    for i in range(0, len(heaps)):
        # print('-' * 10, i, '-' * 10)
        for p in heapq.nlargest(3, heaps[i]):
            # print('Best distance:', p.best_distance)
            words = [TEXT.vocab.itos[j] for j in list(p.X)]

            max_d = max(p.distances)
            min_d = min(p.distances)
            weights = [max_d] + list(p.distances) + [max_d]

            weights = list(1 - (np.array(weights) - min_d) / (max_d - min_d))
            highlighted_text = []
            for word, weight in zip(words, weights):
                highlighted_text.append(
                    f'<span style="background-color:rgba(135,206,250,{str(weight)});">' + html_escape(word) + '</span>')

            highlighted_text = ' '.join(highlighted_text)
            result += highlighted_text + '\n'

    with open("filename.html", "w") as f:
        f.write(result)

if __name__ == '__main__':
    visualize()
