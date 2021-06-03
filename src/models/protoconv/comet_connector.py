import os
from ast import literal_eval
from io import BytesIO

import comet_ml
import pandas as pd

from embeddings_dataset_utils import get_dataset
from models.protoconv.lit_module import ProtoConvLitModule


class CometConnector:
    def __init__(self, apikey, project_name, workspace):
        self.comet_api = comet_ml.api.API(api_key=apikey)
        self.apikey = apikey
        self.project_name = project_name
        self.workspace = workspace

        self.experiment = None
        self.dataset = None
        self.TEXT, self.LABEL, self.train_loader, self.val_loader = None, None, None, None
        self.model = None

    def set_experiment(self, experiment_id, fold=0):
        self.experiment = self.comet_api.get(project_name=self.project_name, workspace=self.workspace,
                                             experiment=experiment_id)
        self.dataset = self.experiment.get_parameters_summary('data_set')['valueCurrent']
        kfold_split_id = list(filter(
            lambda x: x['fileName'] == 'kfold_split_indices.csv', self.experiment.get_asset_list())
        )[0]['assetId']
        kfold_split_binary = self.experiment.get_asset(kfold_split_id, return_type="binary")
        kfold_split = pd.read_csv(BytesIO(kfold_split_binary)).iloc[fold]

        train_index = literal_eval(kfold_split['train_indices'])
        val_index = literal_eval(kfold_split['val_indices'])
        test_index = literal_eval(kfold_split['test_indices'])

        df_dataset = pd.read_csv(f'data/{self.dataset}/tokenized_data.csv')
        train_df, valid_df = df_dataset.iloc[train_index + val_index], df_dataset.iloc[test_index]

        self.TEXT, self.LABEL, self.train_loader, self.val_loader = get_dataset(train_df, valid_df, batch_size=1,
                                                                                cache=None)

    def get_model(self, weights_path):
        if not os.path.isfile('checkpoints/' + weights_path):
            self.experiment.download_model(name=weights_path, output_path='checkpoints/', expand=True)
        self.model = ProtoConvLitModule.load_from_checkpoint('checkpoints/' + weights_path)
