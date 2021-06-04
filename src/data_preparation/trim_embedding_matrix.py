"""
This script removes embeddings that are not used in any dataset
"""
import string

import pandas as pd
import torch
from nltk.corpus import stopwords
from torchtext import data
from torchtext.vocab import _infer_shape
from tqdm import tqdm

from dataframe_dataset import DataFrameDataset

set_of_words = set()
for dataset in ['imdb', 'amazon', 'yelp', 'rottentomatoes', 'hotel']:
    # The settings has to be the same like in src/embeddings_dataset_utils.py
    TEXT = data.Field(init_token='<START>', eos_token='<END>', tokenize=None, tokenizer_language='en',
                      batch_first=True, lower=True,
                      stop_words=set(stopwords.words('english')) | set(string.punctuation)
                      )
    LABEL = data.Field(dtype=torch.float, is_target=True, unk_token=None, sequential=False, use_vocab=False)

    df_dataset = pd.read_csv(f'data/{dataset}/toknized_data.csv')
    entire_dataset = DataFrameDataset(df_dataset, {
        'text': TEXT,
        'label': LABEL
    })

    for example in tqdm(entire_dataset.examples):
        set_of_words.update(example.text)
print(len(set_of_words))

path = '.vector_cache/wiki.en.vec'
path_pt = '.vector_cache/wiki2.en.vec.pt'

vectors_loaded = 0
with open(path, 'rb') as f:
    num_lines, dim = _infer_shape(f)
    max_vectors = num_lines

    itos, vectors, dim = [], torch.zeros((max_vectors, dim)), None

    for line in tqdm(f, total=max_vectors):
        # Explicitly splitting on " " is important, so we don't
        # get rid of Unicode non-breaking spaces in the vectors.
        entries = line.rstrip().split(b" ")

        word, entries = entries[0], entries[1:]
        if dim is None and len(entries) > 1:
            dim = len(entries)
        elif len(entries) == 1:
            print("Skipping token {} with 1-dimensional "
                  "vector {}; likely a header".format(word, entries))
            continue
        elif dim != len(entries):
            raise RuntimeError(
                "Vector for token {} has {} dimensions, but previously "
                "read vectors have {} dimensions. All vectors must have "
                "the same number of dimensions.".format(word, len(entries),
                                                        dim))

        try:
            if isinstance(word, bytes):
                word = word.decode('utf-8')
        except UnicodeDecodeError:
            print("Skipping non-UTF8 token {}".format(repr(word)))
            continue

        if word in set_of_words:
            vectors[vectors_loaded] = torch.tensor([float(x) for x in entries])
            vectors_loaded += 1
            itos.append(word)

print(vectors_loaded)
itos = itos
stoi = {word: i for i, word in enumerate(itos)}
vectors = torch.Tensor(vectors).view(-1, dim)
dim = dim
print('Saving vectors to {}'.format(path_pt))
torch.save((itos, stoi, vectors, dim), path_pt)
