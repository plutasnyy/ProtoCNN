import pandas as pd
from sklearn.model_selection import train_test_split

pd.set_option('display.max_colwidth', 250)

df = pd.read_csv('../data/rottentomatoes/clean/rotten_tomatoes_critic_reviews.csv')
df = df[['review_content', 'review_type']]
df['label'] = (df['review_type'] == 'Fresh').astype(int)
df = df.drop(['review_type'], axis=1)
df = df.dropna()

df_pos = df[df['label'] == 1].sample(n=30000)
df_neg = df[df['label'] == 0].sample(n=30000)

result_df = pd.concat([df_pos, df_neg]).sample(frac=1)
train_df, test_df = train_test_split(result_df, test_size=0.5, stratify=result_df['label'])

print(train_df.head(10))
print(len(train_df))
print(train_df['label'].value_counts())

train_df.to_csv('../data/rottentomatoes/train.csv', index=False)
test_df.to_csv('../data/rottentomatoes/test.csv', index=False)
