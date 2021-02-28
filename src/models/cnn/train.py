import os

os.environ['COMET_DISABLE_AUTO_LOGGING'] = '1'
from configparser import ConfigParser
from copy import deepcopy
from pathlib import Path

import click
import pandas as pd
from easydict import EasyDict

from models.cnn.dataframe_dataset import DataFrameDataset
from models.cnn.lit_module import CNNLitModule
from utils import get_n_splits, log_splits

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.loggers.base import DummyLogger

from torchtext import data
from torchtext.data import BucketIterator
from torchtext.vocab import FastText
import numpy as np


@click.command()
@click.option('-n', '--name', required=True, type=str)
@click.option('-ds', '--data-set', required=True,
              type=click.Choice(['amazon', 'hotel', 'imdb', 'yelp', 'rottentomatoes']))
@click.option('--logger/--no-logger', default=True)
@click.option('-e', '--epoch', default=30, type=int)
@click.option('-f', '--fold', default=1, type=int)
@click.option('-lr', default=2e-5, type=float)
@click.option('--find-lr', default=False, is_flag=True)
@click.option('--test', default=False, is_flag=True)
@click.option('--seed', default=0, type=int)
@click.option('-bs', '--batch-size', default=32, type=int)
@click.option('-fdr', '--fast-dev-run', default=False, type=int)
@click.option('--cache', default=None, type=str)
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
            save_weights_only=True, save_top_k=3,
            monitor=f'val_acc_{i}', period=1
        )
        early_stop = EarlyStopping(monitor=f'val_loss_{i}', patience=5, verbose=True, mode='min', min_delta=0.005)
        callbacks = deepcopy(base_callbacks) + [model_checkpoint, early_stop]

        if params.test:
            print('TESTING')
            train_df, valid_df = df_dataset.iloc[train_index + val_index], df_dataset.iloc[test_index]
        else:
            print('VALIDATING')
            train_df, valid_df = df_dataset.iloc[train_index], df_dataset.iloc[val_index]

        TEXT = data.Field(init_token='<START>', eos_token='<END>', tokenize=None, tokenizer_language='en',
                          batch_first=True, lower=True)
        LABEL = data.Field(dtype=torch.float, is_target=True, unk_token=None, sequential=False)

        train_dataset = DataFrameDataset(train_df, {
            'text': TEXT,
            'label': LABEL
        })

        val_dataset = DataFrameDataset(valid_df, {
            'text': TEXT,
            'label': LABEL
        })

        train_loader, val_loader = BucketIterator.splits(
            (train_dataset, val_dataset),
            batch_size=params.batch_size,
            sort_key=lambda x: len(x.text),
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        TEXT.build_vocab(train_dataset.text, vectors=FastText('en', cache=params.cache))
        LABEL.build_vocab(train_dataset.label)

        model = CNNLitModule(vocab_size=len(TEXT.vocab), embedding_dim=TEXT.vocab.vectors.shape[1], lr=params.lr,
                             fold_id=fold_id)
        model.embedding.weight.data.copy_(TEXT.vocab.vectors)

        trainer = Trainer(
            auto_lr_find=params.find_lr,
            logger=logger,
            max_epochs=params.epoch,
            callbacks=callbacks,
            gpus=1,
            deterministic=True,
            fast_dev_run=params.fast_dev_run,
        )
        trainer.tune(model, train_dataloader=train_loader, val_dataloaders=val_loader)
        trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)

        for absolute_path in model_checkpoint.best_k_models.keys():
            logger.experiment.log_model(Path(absolute_path).name, absolute_path)
        if model_checkpoint.best_model_score:
            best_models_scores.append(model_checkpoint.best_model_score.tolist())
            logger.log_metrics({'best_model_score_' + i: model_checkpoint.best_model_score.tolist()})

    if len(best_models_scores) >= 1:
        logger.log_metrics({
            'avg_best_scores': float(np.mean(np.array(best_models_scores))),
            'std_best_scores': float(np.std(np.array(best_models_scores))),
        })


if __name__ == '__main__':
    train()
