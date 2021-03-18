from ast import literal_eval
from configparser import ConfigParser
from io import BytesIO

import click
import comet_ml
import torch
from easydict import EasyDict

import pandas as pd

from models.protoconv.lit_module import ProtoConvLitModule
from models.protoconv.train import get_dataset


@click.command()
@click.option('--experiment', required=True, type=str, help='For example ce132011516346c99185d139fb23c70c')
@click.option('--weights-path', required=True, type=str, help='For example epoch=25-val_mae=8.2030.ckpt')
@click.option('--fold', required=True, type=int)
def visualize(experiment, weights_path, fold):
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
    model = ProtoConvLitModule.load_from_checkpoint('checkpoints/'+weights_path, vocab_size=len(TEXT.vocab),
                                                    embedding_dim=TEXT.vocab.vectors.shape[1], lr=1, fold_id=fold)
    model.eval()
    with torch.no_grad():
        for i in train_loader:
            print(i)
            break

if __name__ == '__main__':
    visualize()
