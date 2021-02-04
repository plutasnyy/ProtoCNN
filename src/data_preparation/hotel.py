import pandas as pd
from sklearn.model_selection import train_test_split

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

train_df, test_df = train_test_split(result_df, test_size=0.5, stratify=result_df['label'])

print(train_df.head())
print(len(train_df))
print(train_df['label'].value_counts())

train_df.to_csv('../data/hotel/train.csv', index=False)
test_df.to_csv('../data/hotel/test.csv', index=False)
