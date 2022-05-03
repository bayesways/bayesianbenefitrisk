import pandas as pd
import numpy as np
import altair as alt

def form_df(samples, rows):
    dfs = []
    for r in range(rows):
        if rows>1:
            df = pd.DataFrame(samples[:,r])
        else:
            df = pd.DataFrame(samples)
        df.insert(0, 'idx', np.arange(df.shape[0]))
        df = df.melt(id_vars ='idx', var_name = 'col')
        df.insert(1, 'row' , r)
        dfs.append(df)
    return pd.concat(dfs).reset_index(drop=True)

def get_post_df(samples):
    if samples.ndim > 2:
        rows = samples.shape[1]
        df = form_df(samples, rows)
    else:
        rows = 1
        df = form_df(samples, 1)
    return df

def plot_density(df, width=300, height=50) :
    c = alt.Chart(df).transform_fold(
        ['value']
        ).transform_density(
            density='value',
            groupby=['row', 'col', 'source'],
        ).mark_area(opacity=0.5).encode(
            alt.X('value:Q', title=None),
            alt.Y('density:Q'),
            alt.Row('row'),
            alt.Column('col'),
            alt.Color('source')
        ).resolve_scale(
            x='independent'
        ).properties(width=width, height=height)
    return c

def plot_histogram(df, opacity = 0.5, width=300, height=50) :
    c = alt.Chart(df).mark_bar(
        opacity=opacity
    ).encode(
        alt.X('value:Q', bin=alt.Bin(maxbins=100)),
        alt.Y('count()'), 
        alt.Color('source'),
        alt.Row('row'),
        alt.Column('col')
    ).resolve_scale(
        x='independent'
    ).properties(width=width, height=height)
    return c
        
def plot_line(df, width=300, height=50) :
    # it only works for one chain for now
    c = alt.Chart(df).mark_line(
        strokeWidth = 1,
        ).encode(
        alt.X('idx:Q', title=None),
        alt.Y('value:Q'),
        alt.Row('row'),
        alt.Column('col'),
        alt.Color('source')
        ).resolve_scale(
            x='independent'
        ).properties(width=width, height=height)
    return c

def plot_correlations(samples, width=300, height=50) :
    corrdf = pd.DataFrame(samples, columns=['corr'])
    corrdf['idx'] = np.arange(corrdf.shape[0])
    c = alt.Chart(corrdf).mark_bar().encode(
            alt.X('idx:Q', title='data iteration t'),
            alt.Y('corr:Q')
            ).properties(width=width, height=height)
    return c
