from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib.pyplot import savefig, clf

sns.set_style('darkgrid')

result_df = pd.DataFrame()
Path('img/distribution').mkdir(exist_ok=True, parents=True)

for d in ['amazon', 'hotel', 'imdb', 'rottentomatoes', 'yelp']:
    df = pd.read_csv('data/{}/data.csv'.format(d))
    df['length'] = df['text'].apply(lambda x: len(x.split()))
    df['length'].plot.hist(title=f'Number of tokens in input example - {d}')
    savefig(f'img/distribution/{d}.jpg')
    clf()
