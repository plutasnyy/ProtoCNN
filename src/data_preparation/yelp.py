import pandas as pd

pd.set_option('display.max_colwidth', 250)

for part in ['train']:
    df = pd.read_csv('../data/yelp/clean/{}.csv'.format(part), names=['stars', 'text'])
    df['label'] = (df['stars'] >= 3).astype(int)

    df_pos = df[df['label'] == 1].sample(n=15000)
    df_neg = df[df['label'] == 0].sample(n=15000)

    result_df = pd.concat([df_pos, df_neg]).sample(frac=1)
    result_df = result_df.drop(['stars'], axis=1)

    print(result_df.head())
    print(len(result_df))
    print(result_df['label'].value_counts())

    result_df.to_csv('../data/yelp/data.csv', index=False)
