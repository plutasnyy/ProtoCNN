import os

from torchtext.vocab import FastText

os.environ['COMET_DISABLE_AUTO_LOGGING'] = '1'

from configparser import ConfigParser
from copy import deepcopy
from pathlib import Path

import click
import pandas as pd

from click_option_group import optgroup
from easydict import EasyDict
from utils import get_n_splits, log_splits

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.loggers.base import DummyLogger

from models.protoconv.visualize_prototypes import visualize_model
from configs import dataset_tokens_length, model_to_litmodule, dataset_to_number_of_prototypes

import numpy as np


@click.command()
@optgroup.group('COMET ML conf', help='The configuration of logging connection')
@optgroup.option('--run-name', required=True, type=str)
@optgroup.option('--project-name', default=None, type=str, help='Override project name from config.ini')
@optgroup.option('--cache', default=None, type=str, help='Path to the cached embeddings, None for local runs and '
                                                         'drive:/.vectors_cached in order to run in the cloud')
@optgroup.option('--logger/--no-logger', default=True, help='Turn off logging for local runs')
@optgroup.group('TRAINING conf')
@optgroup.option('--model', type=click.Choice(['distilbert', 'cnn', 'protoconv']), required=True,
                 help='Which model should be used')
@optgroup.option('--data-set', required=True,
                 type=click.Choice(['amazon', 'hotel', 'imdb', 'yelp', 'rottentomatoes']))
@optgroup.option('--epoch', default=30, type=int, help='Number of epochs')
@optgroup.option('--fold', default=1, type=int, help='Whenever train using one split, or 5-fold')
@optgroup.option('-lr', default=2e-5, type=float, help='Learning rate')
@optgroup.option('--find-lr', default=False, is_flag=True,
                 help='Automatic finding learning rate, usually doesnt work')
@optgroup.option('--batch-size', default=32, type=int, help='Batch size')
@optgroup.option('--seed', default=0, type=int)
@optgroup.option('-fdr', '--fast-dev-run', default=False, type=int, help='Train/valid/test 1 epoch with only N batches')
@optgroup.group('TRANSFORMER conf')
@optgroup.option('--tokenizer-length', default=None, type=int,
                 help='Max length of input for tokenizer, if None use value from dataset configuration')
@optgroup.group('CNN conf')
@optgroup.option('--cnn-conv-filters', default=32, type=int, help='Number of convolutional filters')
@optgroup.option('--cnn-filter-size', default=3, type=int, help='Size of convolutional filter')
@optgroup.group('PROTOCONV conf')
@optgroup.option('--pc-sim-func', required=False, type=click.Choice(['linear', 'log']), default='log',
                 help='Function used to change distance to similarity, linear is -d, log is lod((d+1)/d')
@optgroup.option('--pc-separation-threshold', default=0, type=float,
                 help='After that distance the seperation cost is ignored, optimal value between 0-2')
@optgroup.option('--pc-cls-loss-weight', default=0, type=float, help='Weight of clustering loss')
@optgroup.option('--pc-sep-loss-weight', default=0, type=float, help='Weight of separation loss')
@optgroup.option('--pc-number-of-prototypes', default=None, type=int,
                 help='Number of prototypes, if None or -1 the default value from configs.py '
                      'for each dataset will be chosen')
@optgroup.option('--pc-conv-filters', default=32, type=int,
                 help='Number of convolutional filters, also size of the prototype')
@optgroup.option('--pc-filter-size', default=3, type=int, help='Size of convolutional filter')
@optgroup.option('--pc-stride', default=1, type=int, help='Size of stride in convolutional layer')
@optgroup.option('--pc-project-prototypes-every-n', default=4, type=int)
@optgroup.option('--pc-prototypes-init', type=click.Choice(['rand', 'zeros', 'xavier']), default='rand',
                 help='How weights in PrototypeLayer should be initialized')
@optgroup.option('--pc-visualize', default=False, help='Visualize prototypes after best model in first fold. '
                                                       'Used in trainer and fabric of ProtoConv Module')
def train(**args):
    params = EasyDict(args)
    seed_everything(params.seed)

    config = ConfigParser()
    config.read('config.ini')

    if params.model == 'distilbert' and params.tokenizer_length is None:
        params.tokenizer_length = dataset_tokens_length[params.data_set]

    if params.model == 'protoconv' and (params.pc_number_of_prototypes is None or params.pc_number_of_prototypes == -1):
        params.pc_number_of_prototypes = dataset_to_number_of_prototypes[params.data_set]

    logger = DummyLogger()
    if params.logger:
        comet_config = EasyDict(config['cometml'])
        project_name = params.project_name if params.project_name else comet_config.projectname
        logger = CometLogger(api_key=comet_config.apikey, project_name=project_name, workspace=comet_config.workspace)

    # logger.experiment.log_code(folder='src')
    logger.log_hyperparams(params)
    base_callbacks = [LearningRateMonitor(logging_interval='epoch')]

    df_dataset = pd.read_csv(f'data/{params.data_set}/data.csv')
    n_splits = get_n_splits(dataset=df_dataset, x_label='text', y_label='label', folds=params.fold)
    # log_splits(n_splits, logger)

    embeddings = FastText('en', cache=params.cache) if params.model != 'distilbert' else None

    best_models_scores = []
    for fold_id, (train_index, val_index, test_index) in enumerate(n_splits):
        i = str(fold_id)
        model_checkpoint = ModelCheckpoint(
            filepath='checkpoints/fold_' + i + '_{epoch:02d}-{val_loss_' + i + ':.4f}-{val_acc_' + i + ':.4f}',
            save_weights_only=True, save_top_k=1, monitor='val_acc_' + i, period=1
        )
        early_stop = EarlyStopping(monitor=f'val_loss_{i}', patience=7, verbose=True, mode='min', min_delta=0.005)
        callbacks = deepcopy(base_callbacks) + [model_checkpoint, early_stop]
        train_df, valid_df = df_dataset.iloc[train_index + val_index], df_dataset.iloc[test_index]

        lit_module = model_to_litmodule[params.model]
        model, train_loader, val_loader = lit_module.from_params_and_dataset(train_df, valid_df, params, fold_id,
                                                                             embeddings)

        trainer = Trainer(auto_lr_find=params.find_lr, logger=logger, max_epochs=params.epoch, callbacks=callbacks,
                          gpus=1, deterministic=True, fast_dev_run=params.fast_dev_run)
        trainer.tune(model, train_dataloader=train_loader, val_dataloaders=val_loader)
        trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)

        for absolute_path in model_checkpoint.best_k_models.keys():
            logger.experiment.log_model(Path(absolute_path).name, absolute_path)

        if model_checkpoint.best_model_score:
            best_models_scores.append(model_checkpoint.best_model_score.tolist())
            logger.log_metrics({'best_model_score_' + i: model_checkpoint.best_model_score.tolist()}, step=0)

        if params.model == 'protoconv' and model_checkpoint.best_model_path and params.pc_visualize and fold_id == 0:
            best_model = lit_module.load_from_checkpoint(model_checkpoint.best_model_path)
            visualization_path = f'prototypes_visualization_{fold_id}.html'
            visualize_model(best_model, train_loader, k_most_similar=3, file_name=visualization_path)
            logger.experiment.log_asset(visualization_path)

    if len(best_models_scores) >= 1:
        avg_best, std_best = float(np.mean(np.array(best_models_scores))), float(np.std(np.array(best_models_scores)))
        table_entry = f'{avg_best:.3f} ($\pm${std_best:.3f})'

        logger.log_metrics({
            'avg_best_scores': avg_best,
            'std_best_scores': std_best,
        })
        logger.log_hyperparams({'table_entry': table_entry})


if __name__ == '__main__':
    train()
