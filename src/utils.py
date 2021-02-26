from pathlib import Path

import dill
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torchtext.data import Dataset


def call_click_wrapper(f, run_params: dict):
    list_of_params = []
    for k, v in run_params.items():
        list_of_params.extend([k, v])
    f(list_of_params)


def get_n_splits(dataset, x_label, y_label, folds):
    """
    split with ratio train/val/test 60/20/20
    :param folds: if >
    :return: list of indices in splits [(train_id, test_id)]
    """
    if folds == 1:
        train_indices, test_indices = train_test_split(list(range(len(dataset))), test_size=0.2,
                                                       stratify=dataset[y_label])
        n_splits = [(train_indices, test_indices)]
    else:
        skf = StratifiedKFold(n_splits=folds)
        n_splits = list(skf.split(X=dataset[x_label], y=dataset[y_label]))

    result = []
    for train_indices, test_indices in n_splits:
        train_ids, val_ids = train_test_split(train_indices, test_size=0.25)
        result.append((train_ids, val_ids, test_indices))

    return result


def log_splits(n_splits, logger):
    df = pd.DataFrame(n_splits, columns=['train_indices', 'val_indices', 'test_indices'])
    logger.experiment.log_table('kfold_split_indices.csv', tabular_data=df)


def save_torchtext_dataset(dataset, path):
    if not isinstance(path, Path):
        path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(dataset.examples, path / "examples.pkl", pickle_module=dill)
    torch.save(dataset.fields, path / "fields.pkl", pickle_module=dill)


def load_torchtext_dataset(path):
    if not isinstance(path, Path):
        path = Path(path)
    examples = torch.load(path / "examples.pkl", pickle_module=dill)
    fields = torch.load(path / "fields.pkl", pickle_module=dill)
    return Dataset(examples, fields)


def get_pad_to_min_len_fn(min_length):
    def pad_to_min_len(batch, vocab, min_length=min_length):
        pad_idx = vocab.stoi['<pad>']
        for idx, ex in enumerate(batch):
            if len(ex) < min_length:
                batch[idx] = ex + [pad_idx] * (min_length - len(ex))
        return batch

    return pad_to_min_len
