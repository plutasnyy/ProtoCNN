import os.path
import warnings
from ast import literal_eval
from configparser import ConfigParser
from io import BytesIO

import click
import comet_ml
import pandas as pd
from easydict import EasyDict
from pytorch_lightning import seed_everything

from embeddings_dataset_utils import get_dataset
from models.protoconv.data_visualizer import DataVisualizer

warnings.simplefilter("ignore")


@click.command()
@click.option('--experiment', required=True, type=str, help='For example d95670fd157f41eca8c0f8b97f8ffd91')
@click.option('--weights-path', required=True, type=str, help='For example fold_0_epoch=17-val_loss_0=0.4810-val_acc_0=0.7957.ckpt')
@click.option('--fold', required=True, type=int)
def visualize(experiment, weights_path, fold):
    seed_everything(0)

    config = ConfigParser()
    config.read('config.ini')
    comet_config = EasyDict(config['cometml'])
    comet_api = comet_ml.api.API(api_key=comet_config.apikey)
    experiment = comet_api.get(project_name=comet_config.projectname, workspace=comet_config.workspace,
                               experiment=experiment)

    if not os.path.isfile('checkpoints/' + weights_path):
        experiment.download_model(name=weights_path, output_path='checkpoints/', expand=True)

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

    data_visualizer = DataVisualizer(model, train_loader, vocab_itos=TEXT.vocab.itos)
    text = "This movie was absolutely terrible, I've never watched anything so boring before. The good thing was that it ended quickly."
    tokenized = TEXT.process([TEXT.preprocess(text)]).cuda()
    data_visualizer.predict(tokenized, 0, output_file_path='explanation.html')
    # data_visualizer.visualize_prototypes_as_bold(output_file_path='prototypes1.html', short=True)
    # data_visualizer.visualize_prototypes_as_heatmap(output_file_path='prototypes2.html')
    # data_visualizer.visualize_prototypes_as_heatmap(output_file_path='prototypes.html')


if __name__ == '__main__':
    visualize()
