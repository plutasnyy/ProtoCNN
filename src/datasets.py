import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SentimentDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, length):
        self.df = df
        self.tokenizer = tokenizer
        self.length = length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        encoded = self.tokenizer(row['text'], add_special_tokens=True, padding='max_length', truncation=True,
                                 return_offsets_mapping=True, max_length=self.length)
        encoded['labels'] = np.array([
            1 if any(left <= chr_pos < right for chr_pos in row['spans']) else 0
            for left, right in encoded['offset_mapping']
        ])

        encoded['sentence_id'] = row['sentence_id']
        encoded['offset'] = row['offset']

        item = {k: torch.tensor(v).long() for k, v in encoded.items()}
        return item
