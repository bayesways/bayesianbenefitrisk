import numpy as np
from scipy.stats import bernoulli, multivariate_normal
from scipy.special import expit, logsumexp


def gen_weights_master(
    data,
    particles,
    M
    ):
    """
    dim(y) = k 

    For a single data point y, compute the
    likelihood of each particle as the 
    average likelihood across a sample of latent
    variables z.
    """

    weights = np.empty(M)
    for m in range(M):
        weights[m] = multivariate_normal.logpdf(
            x = data['y'],
            mean = particles['alpha'][m].reshape(
                data['y'].shape
                ),
            cov = particles['Marg_cov'][m].reshape(
                data['y'].shape[0],
                data['y'].shape[0]
                ),
            allow_singular=True
        )
    return weights
