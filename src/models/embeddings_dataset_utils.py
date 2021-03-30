import torch
from torchtext import data
from torchtext.data import BucketIterator
from torchtext.vocab import FastText

from models.dataframe_dataset import DataFrameDataset


def get_dataset(train_df, valid_df, batch_size, cache, gpus=1):
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
        batch_size=batch_size,
        sort_key=lambda x: len(x.text),
        device='cuda' if torch.cuda.is_available() and gpus else 'cpu'
    )

    TEXT.build_vocab(train_dataset.text, vectors=FastText('en', cache=cache))
    LABEL.build_vocab(train_dataset.label)

    return TEXT, LABEL, train_loader, val_loader
