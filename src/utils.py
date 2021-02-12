import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def call_click_wrapper(f, run_params: dict):
    list_of_params = []
    for k, v in run_params.items():
        list_of_params.extend([k, v])
    f(list_of_params)


def get_n_splits(dataset, x_label, y_label, test_size, folds):
    """
    :param test_size: used when folds == 1
    :param folds: if > 1 ignore test_size
    :return: list of indices in splits [(train_id, test_id)]
    """
    if folds == 1:
        train_indices, val_indices = train_test_split(list(range(len(dataset))), test_size=test_size,
                                                      stratify=dataset[y_label])
        n_splits = [(train_indices, val_indices)]
    else:
        skf = StratifiedKFold(n_splits=folds)
        n_splits = list(skf.split(X=dataset[x_label], y=dataset[y_label]))
    return n_splits


def log_splits(n_splits, logger):
    df = pd.DataFrame(n_splits, columns=['train_indices', 'test_indices'])
    logger.experiment.log_table('kfold_split_indices.csv', tabular_data=df)
