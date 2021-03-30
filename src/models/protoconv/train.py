import os

os.environ['COMET_DISABLE_AUTO_LOGGING'] = '1'

from models.embeddings_dataset_utils import get_dataset

from configparser import ConfigParser
from copy import deepcopy
from pathlib import Path

import click
import pandas as pd
from easydict import EasyDict

from utils import get_n_splits, log_splits

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.loggers.base import DummyLogger

from models.protoconv.lit_module import ProtoConvLitModule
from models.protoconv.visualize_prototypes import visualize_model

import numpy as np


@click.command()
@click.option('--run-name', required=True, type=str)
@click.option('--data-set', required=True,
              type=click.Choice(['amazon', 'hotel', 'imdb', 'yelp', 'rottentomatoes']))
@click.option('--logger/--no-logger', default=True)
@click.option('--epoch', default=30, type=int)
@click.option('--fold', default=1, type=int)
@click.option('-lr', default=1e-3, type=float)
@click.option('--find-lr', default=False, is_flag=True)
@click.option('--test', default=False, is_flag=True)
@click.option('--seed', default=0, type=int)
@click.option('--batch-size', default=32, type=int)
@click.option('-fdr', default=False, type=int)
@click.option('--cache', default=None, type=str)
@click.option('--sim-func', required=False, type=click.Choice(['linear', 'log']), default='log')
@click.option('--separation-threshold', default=0, type=float)
@click.option('--cls-loss-weight', default=0, type=float)
@click.option('--sep-loss-weight', default=0, type=float)
@click.option('--number-of-prototypes', default=16, type=int)
@click.option('--latent-size', default=32, type=int)
def train(**params):
    params = EasyDict(params)
    seed_everything(params.seed)

    config = ConfigParser()
    config.read('config.ini')

    logger = DummyLogger()
    if params.logger:
        comet_config = EasyDict(config['cometml'])
        logger = CometLogger(api_key=comet_config.apikey, project_name=comet_config.projectname,
                             workspace=comet_config.workspace)

    logger.experiment.log_code(folder='src')
    logger.log_hyperparams(params)
    base_callbacks = [LearningRateMonitor(logging_interval='epoch')]

    df_dataset = pd.read_csv(f'data/{params.data_set}/data.csv')
    n_splits = get_n_splits(dataset=df_dataset, x_label='text', y_label='label', folds=params.fold)
    log_splits(n_splits, logger)

    best_models_scores = []
    for fold_id, (train_index, val_index, test_index) in enumerate(n_splits):
        i = str(fold_id)
        model_checkpoint = ModelCheckpoint(
            filepath='checkpoints/fold_' + i + '_{epoch:02d}-{val_loss_' + i + ':.4f}-{val_acc_' + i + ':.4f}',
            save_weights_only=False, save_top_k=3, monitor=f'val_acc_{i}', period=1
        )
        early_stop = EarlyStopping(monitor=f'val_loss_{i}', patience=100, verbose=True, mode='min', min_delta=0.005)
        callbacks = deepcopy(base_callbacks) + [model_checkpoint, early_stop]

        train_df, valid_df = df_dataset.iloc[train_index + val_index], df_dataset.iloc[test_index]

        TEXT, LABEL, train_loader, val_loader = get_dataset(train_df, valid_df, params.batch_size, params.cache, gpus=1)

        model = ProtoConvLitModule(vocab_size=len(TEXT.vocab), embedding_dim=TEXT.vocab.vectors.shape[1],
                                   fold_id=fold_id, **params)
        model.embedding.weight.data.copy_(TEXT.vocab.vectors)

        trainer = Trainer(auto_lr_find=params.find_lr, logger=logger, max_epochs=params.epoch, callbacks=callbacks,
                          gpus=1, deterministic=True, fast_dev_run=params.fdr)
        trainer.tune(model, train_dataloader=train_loader, val_dataloaders=val_loader)
        trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)

        for absolute_path in model_checkpoint.best_k_models.keys():
            logger.experiment.log_model(Path(absolute_path).name, absolute_path)

        if model_checkpoint.best_model_score:
            best_models_scores.append(model_checkpoint.best_model_score.tolist())
            logger.log_metrics({'best_model_score_' + i: model_checkpoint.best_model_score.tolist()})

        if model_checkpoint.best_model_path:
            if fold_id == 0:
                best_model = ProtoConvLitModule.load_from_checkpoint(model_checkpoint.best_model_path)
                visualization_path = f'prototypes_visualization_{fold_id}.html'
                visualize_model(best_model, train_loader, k_most_similar=3, file_name=visualization_path,
                                vocab_int_to_string=TEXT.vocab.itos)
                logger.experiment.log_asset(visualization_path)

    if len(best_models_scores) >= 1:
        logger.log_metrics({
            'avg_best_scores': float(np.mean(np.array(best_models_scores))),
            'std_best_scores': float(np.std(np.array(best_models_scores))),
        })


if __name__ == '__main__':
    train()
