import pandas as pd

pd.set_option('display.max_colwidth', 250)

df1 = pd.read_csv('../data/hotel/clean/Datafiniti_Hotel_Reviews.csv')
df2 = pd.read_csv('../data/hotel/clean/Datafiniti_Hotel_Reviews_Jun19.csv')

df = pd.concat([df1, df2])
df = df[['reviews.rating', 'reviews.text']]
df.columns = ['label', 'text']
df['label'] = df['label'].astype(int)
df = df[df['label'] != 3]
df['label'] = (df['label'] > 3).astype(int)

df_neg = df[df['label'] == 0]
df_pos = df[df['label'] == 1].sample(n=2374)
result_df = pd.concat([df_pos, df_neg]).sample(frac=1)

print(result_df.head())
print(len(result_df))
print(result_df['label'].value_counts())

result_df.to_csv('../data/hotel/data.csv', index=False)
