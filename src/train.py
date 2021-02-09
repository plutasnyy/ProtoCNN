import os
from configparser import ConfigParser
from pathlib import Path

import click
from dataset import DatasetModule
from easydict import EasyDict
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CometLogger

from configs import model_data
from models.transformer_lit import TransformerLitModule

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['COMET_DISABLE_AUTO_LOGGING'] = '1'


@click.command()
@click.option('-n', '--name', required=True, type=str)
@click.option('-dp', '--data-path', required=True, type=str)
@click.option('-m', '--model', default='distilbert', type=click.Choice(['distilbert']))
@click.option('-l', '--length', default=512, type=click.Choice([128, 512]))
@click.option('--logger/--no-logger', default=True)
@click.option('-e', '--epochs', default=4, type=int)
@click.option('--lr', default=4.7e-5, type=float)
@click.option('--find-lr', default=False, is_flag=True)
@click.option('--seed', default=0, type=int)
@click.option('-bs', '--batch-size', default=32, type=int)
@click.option('-fdr', '--fast-dev-run', default=False, is_flag=True)
def train(**params):
    params = EasyDict(params)
    seed_everything(params.seed)

    config = ConfigParser()
    config.read('config.ini')

    logger, callbacks = False, list()
    if params.logger:
        comet_config = EasyDict(config['cometml'])
        logger = CometLogger(api_key=comet_config.apikey, project_name=comet_config.projectname,
                             workspace=comet_config.workspace)
        logger.experiment.set_code(filename='project/span_bert/train.py', overwrite=True)  # TODO remove deprecated code
        logger.log_hyperparams(params)
        logger.experiment.log_asset_folder('project/span_bert')
        callbacks.append(LearningRateMonitor(logging_interval='epoch'))

    model_checkpoint = ModelCheckpoint(
        filepath='checkpoints/{epoch:02d}-{val_loss:.4f}-{acc:.4f}',
        save_weights_only=True,
        save_top_k=10,
        monitor='val_loss',
        mode='min',
        period=1
    )
    callbacks.extend([model_checkpoint])

    model_class, tokenizer_class, model_name = model_data[params.model]
    tokenizer = tokenizer_class.from_pretrained(model_name, do_lower_case=True)
    model_backbone = model_class.from_pretrained(model_name, num_labels=2, output_attentions=False,
                                                 output_hidden_states=False)

    model = TransformerLitModule(
        model=model_backbone,
        tokenizer=tokenizer,
        lr=params.lr
    )

    data_module = DatasetModule(
        data_dir=params.data_path,
        tokenizer=tokenizer,
        batch_size=params.batch_size,
        length=params.length,
        augmentation=params.augmentation,
        valintrain=params.valintrain
    )

    trainer = Trainer(
        auto_lr_find=params.find_lr,
        logger=logger,
        max_epochs=params.epochs,
        callbacks=callbacks,
        gpus=1,
        deterministic=True,
        fast_dev_run=params.fast_dev_run
    )

    trainer.fit(model, datamodule=data_module)

    if params.logger:
        for absolute_path in model_checkpoint.best_k_models.keys():
            logger.experiment.log_model(Path(absolute_path).name, absolute_path)
        logger.log_metrics({'best_model_score': model_checkpoint.best_model_score.tolist()})


if __name__ == '__main__':
    train()
