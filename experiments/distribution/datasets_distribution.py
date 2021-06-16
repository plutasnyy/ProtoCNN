from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib.pyplot import savefig, clf
import matplotlib.pyplot as plt
sns.set_style('darkgrid')
sns.set_context('paper')
result_df = pd.DataFrame()
Path('distribution').mkdir(exist_ok=True, parents=True)

for d in ['amazon', 'hotel', 'imdb', 'rottentomatoes', 'yelp']:
    df = pd.read_csv('data/{}/data.csv'.format(d))
    df['Number of tokens'] = df['text'].apply(lambda x: len(x.split()))
    sns.displot(df, x="Number of tokens", kind="kde", bw_adjust=2)
    plt.title(f'Length of the examples (in tokens) - {d.upper()}')
    plt.tight_layout()
    savefig(f'distribution/{d}.jpg')
    clf()
