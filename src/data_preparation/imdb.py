from pathlib import Path

import pandas as pd
pd.set_option('display.max_colwidth', 250)

data_path = '../data/imdb/clean/{}/{}'

for part in ['train', 'test']:
    data = []
    for label in ['pos', 'neg']:
        for file_path in Path(data_path.format(part, label)).glob('*.txt'):
            with open(file_path) as f:
                text = f.read()
            data.append([text, int(label == 'pos')])
    df = pd.DataFrame(data, columns=['text', 'label']).sample(frac=1)
    df.to_csv('../data/imdb/{}.csv'.format(part), index=False)
    print(df.head())
