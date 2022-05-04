import pandas as pd
import numpy as np
from scipy.stats import norm, bernoulli


def produce_sample_dataset(seed=0):
    np.random.seed(seed=seed)

    dfgs = []

    for g in ['MET', 'RSG', 'AVM']:
        values = pd.read_csv("../dat/%s.csv"%g).to_dict()

        n = int(values['subjid'][0])
        dfg = pd.DataFrame(np.zeros((n, 8)),
                    columns=['subjid', 'atrtgrp', 'efhgba1_diff', 'effpg_diff', 'Diarrhoea',
               'Dyspepsia', 'Nausea', 'Vomiting']
                )

        dfg['atrtgrp'] = g


        for colname in ['efhgba1_diff', 'effpg_diff']:

            dfg[colname] = norm.rvs(
                loc=values[colname][1],
                scale=values[colname][2],
                size = int(dfg[colname].shape[0])
            )

        for colname in ['Diarrhoea', 'Dyspepsia', 'Nausea', 'Vomiting']:
            dfg[colname] = bernoulli.rvs(
                p=values[colname][1],
                size = int(values[colname][0])
            )

        dfgs.append(dfg)

    df = pd.concat(dfgs).reset_index(drop=True).round(2)
    df['subjid'] = np.arange(1, df.shape[0]+1)
    
    return df
    
produce_sample_dataset().to_csv("../dat/synthetic_dataset.csv", index=False)
