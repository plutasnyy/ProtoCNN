import pandas as pd

pd.set_option('display.max_colwidth', 250)

df = pd.read_csv('../data/rottentomatoes/clean/rotten_tomatoes_critic_reviews.csv')
df = df[['review_content', 'review_type']]
df['label'] = (df['review_type'] == 'Fresh').astype(int)
df = df.drop(['review_type'], axis=1)
df = df.dropna()

df_pos = df[df['label'] == 1].sample(n=15000)
df_neg = df[df['label'] == 0].sample(n=15000)

result_df = pd.concat([df_pos, df_neg]).sample(frac=1)

print(result_df.head(10))
print(len(result_df))
print(result_df['label'].value_counts())

result_df.to_csv('../data/rottentomatoes/data.csv', index=False)
