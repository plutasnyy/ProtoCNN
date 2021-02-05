from pathlib import Path

import pandas as pd

pd.set_option('display.max_colwidth', 250)

data_path = '../data/imdb/clean/{}/{}'

for part in ['train']:
    data = []
    for label in ['pos', 'neg']:
        for file_path in Path(data_path.format(part, label)).glob('*.txt'):
            with open(file_path) as f:
                text = f.read()
            data.append([text, int(label == 'pos')])
    df = pd.DataFrame(data, columns=['text', 'label']).sample(frac=1)

    print(df.head())
    print(len(df))
    print(df['label'].value_counts())

    df.to_csv('../data/imdb/data.csv'.format(part), index=False)
