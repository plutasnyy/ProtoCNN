import csv
import string

import numpy as np
import pandas as pd
import torch
from torchtext import data
from tqdm.contrib import tenumerate

from dataframe_dataset import DataFrameDataset

for dataset in ['imdb', 'amazon', 'yelp', 'rottentomatoes', 'hotel']:
    TEXT = data.Field(init_token='<START>', eos_token='<END>', tokenize='spacy', tokenizer_language='en',
                      batch_first=True, lower=True, stop_words=set(string.punctuation))
    LABEL = data.Field(dtype=torch.float, is_target=True, unk_token=None, sequential=False)

    df_dataset = pd.read_csv(f'data/{dataset}/data.csv')
    entire_dataset = DataFrameDataset(df_dataset, {
        'text': TEXT,
        'label': LABEL
    })

    df_dataset = pd.read_csv(f'data/{dataset}/data.csv')

    tokenized_input = []
    for i, example in tenumerate(entire_dataset.examples):
        words = list(example.text)
        if len(words) > 0:
            tokenized_input.append(' '.join(words))
        else:
            tokenized_input.append(None)

    df_dataset['text'] = tokenized_input
    df_dataset = df_dataset.replace(to_replace='None', value=np.nan).dropna()
    df_dataset.to_csv(f'data/{dataset}/tokenized_data.csv', index=False, quoting=csv.QUOTE_NONNUMERIC)
