import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, bernoulli



def scaled_invlogit(x, a, b):
    return (b*np.exp(x)+a)/(1+np.exp(x))


def scaled_logit(x, a, b):
    return np.log((x-a)/(b-x))


def scaled_invlogit_perrow(x):
    """
    Transform a measurement of `alpha` to MCDA scale
    """
    res = x.copy()
    res[2] = scaled_invlogit(x[2], 0.1,.35)
    res[3] = scaled_invlogit(x[3], 0.1,.25)
    res[4] = scaled_invlogit(x[4], 0.1,.2)
    res[5] = scaled_invlogit(x[5], 0.1,.25)

    return res


def pref_score(x, m1, m2 , sign=-1):
    """
    map from the effect range [m1,m2] to
    the score range [0,100]. Sign = 1
    indicates higher effect meaasurements are more
    desirable, otherwise the opposite.
    """
    m = 100./float((m2-m1))
    if sign == 1:
        return -m * m1 + m*x
    else:
        return m * m2 - m*x


def get_scores_perrow(x):
    """
    Return the preference score for a row of measurements
    for only 4 binary and 2 cont
    """
    res =np.empty(6)
    res[0] = pref_score(x[0], -6.,3.,-1)
    res[1] = pref_score(x[1], -15.,7.5,-1)
    res[2] = pref_score(x[2], 0.1,.35, -1)
    res[3] = pref_score(x[3], 0.1,.25, -1)
    res[4] = pref_score(x[4], 0.1,.2, -1)
    res[5] = pref_score(x[5], 0.1,.25, -1)
    return res


def get_final_score_perrow(x, weights=None):
    """
    dot product of measurments and clinical
    weights.
    """
    scores = get_scores_perrow(x)
    if weights is None:
        weights = np.array( [ .592, .118, .089, .178, .018, 0.005])
    else:
        weights = weights.reshape(x.shape)
    return np.dot(scores,weights)


def get_score(x):
    weights = np.array( [ .592, .118, .089, .178, .018, 0.005])
    # weights = np.array([1./6]*6)
    np.testing.assert_approx_equal(1.,np.sum(weights))
    return np.dot(get_scores_perrow(x), weights)


def get_scores_array(pp_samples, mcmc_samples, num_groups):
    scores = np.empty((mcmc_samples, num_groups))
    for i in range(mcmc_samples):
        for j in range(num_groups):
            s = scaled_invlogit_perrow(pp_samples[i, j])
            scores[i, j] = get_score(s)
    return scores
            
            
def get_scores_df(scores_array, group_names):
    df = pd.DataFrame(
        scores_array,
        columns=group_names
    )
    return df


def get_sample_cov_per_group(data, grpnum):
    yy = data['yc'][data['grp']== grpnum]
    return np.round(np.cov(yy, rowvar=False),2)


def get_prob(scoresdf, grp1, grp2):
    return scoresdf[scoresdf[grp1] < scoresdf[grp2]].shape[0] / scoresdf.shape[0] 
