import bz2

import pandas as pd

pd.set_option('display.max_colwidth', 250)

for part in ['train']:
    file_lines = []
    with bz2.BZ2File('../data/amazon/clean/{}.ft.txt.bz2'.format(part)) as f:
        for i, x in enumerate(f.readlines()):
            if i > 50000:
                break
            file_lines.append(x.decode('utf-8'))
    labels = [int(x.startswith('__label__2')) for x in file_lines]
    sentences = [x.split(' ', 1)[1][:-1] for x in file_lines]

    df = pd.DataFrame({
        'text': sentences,
        'label': labels
    })
    df_pos = df[df['label'] == 1].sample(n=15000)
    df_neg = df[df['label'] == 0].sample(n=15000)
    df = pd.concat([df_pos, df_neg]).sample(frac=1)

    print(df.head())
    print(len(df))
    print(df['label'].value_counts())
    df.to_csv('../data/amazon/data.csv', index=False)
