import numpy as np
from numpy.linalg import inv, cholesky
from scipy.stats import bernoulli, multivariate_normal, norm
from scipy.special import expit, logsumexp
from scipy.optimize import minimize
from pdb import set_trace

def check_posdef(S):
    """
    Check that matrix is positive definite
    Inputs
    ============
    - matrix
    """
    try:
        cholesky(S)
        return 1
    except NameError:
        print("\n Error: could not compute cholesky factor of S")
        raise
        return 0

def initialize_bundles(
    size, 
    bundle_size,
    data
    ):
    latent_bundles = dict()
    latent_bundles['z'] = np.zeros((
        size,
        bundle_size,
        data['N'],
        data['K']
        )
    )
    latent_bundles['y'] = np.zeros((
        size,
        bundle_size,
        data['N'],
        data['J']
        )
    )
    return latent_bundles

def initialize_latentvars(
    size, 
    data
    ):
    latentvars = dict()
    latentvars['z'] = np.zeros((
        size,
        data['N'],
        data['K']
        )
    )
    latentvars['y'] = np.zeros((
        size,
        data['N'],
        data['J']
        )
    )
    return latentvars


def get_weight(
    data,
    yy
    ):
    assert data['N'] == 1
    w = bernoulli.logpmf(
            data['D'],
            p=expit(yy)
            ).sum()
    return w 


def get_bundle_weights(
    bundle_size,
    data,
    y_bundle,
    ):
    bundle_w = np.empty(bundle_size)
    for l in range(bundle_size):
        bundle_w[l] = get_weight(data, y_bundle[l])
    return bundle_w


def get_weight_matrix_at_datapoint(
    size,
    bundle_size,
    data,
    yy):
    weights = np.empty((size,bundle_size))
    for m in range(size):
        weights[m] =  get_bundle_weights(
            bundle_size,
            data,
            yy[m])
    return weights


def get_weight_matrix_for_particle(
    bundle_size,
    data,
    yy):
    weights = np.empty(bundle_size)
    for l in range(bundle_size):
        weights[l] =  bernoulli.logpmf(
            data['D'],
            p=expit(yy[l])
            ).sum()
    return weights

def generate_latent_pair(
    Kb,
    alpha,
    beta):
    pair = dict()
    zz = norm.rvs()
    yy = alpha + np.squeeze(np.outer(zz, beta))
    pair['z'] = zz
    pair['y'] = yy
    return pair

def generate_latent_pair_laplace(
    data_y,
    alpha,
    beta):
    lapldist =  get_laplace_approx(
        data_y,
        {
            'alpha':alpha,
            'beta':beta,
        }
        )
    pair = dict()
    zz = lapldist.rvs()
    yy = alpha + np.squeeze(np.outer(zz, beta))
    pair['z'] = zz
    pair['y'] = yy
    return pair


## Laplace Approximation functions
def get_pi_z(z, theta):
    exp_eta = np.exp(theta['alpha'] +  np.squeeze(np.outer(z,theta['beta'])))
    return exp_eta/(1+exp_eta)


def get_log_likelihood(z,y,theta):
    pi_z = get_pi_z(z, theta)
    s1 = (y*np.log(pi_z))+((1.-y)*(np.log(1. - pi_z)))
    return np.sum(s1)

def get_neg_posterior(z,y,theta):
    return -1.*(get_log_likelihood(z,y,theta)+norm.logpdf(z))

def get_grad_pi_z(z, theta):
    exp_eta = np.exp(theta['alpha'] +  np.squeeze(np.outer(z,theta['beta'])))
    return (exp_eta *  theta['beta'].T)/(1+exp_eta)**2

def get_fisher_information(z, y, theta):
    pi_z = get_pi_z(z, theta)
    grad_pi_z = get_grad_pi_z(z, theta)
    r1 =grad_pi_z**2
    r2 =pi_z*(1.-pi_z)
    return 1. + np.sum(r1/r2)

def get_laplace_approx(y, theta):
    res = minimize(get_neg_posterior, np.zeros(1), args=(y, theta), method='BFGS')
    cov_matrix = get_fisher_information(res.x, y, theta).reshape((1,1))
    if check_posdef(cov_matrix) == 0:
        cov_matrix = np.eye(theta['beta'].shape[1])
    return multivariate_normal(mean = res.x, cov = cov_matrix**(-1), allow_singular=True )


# def get_laplace_approx(y, theta):
#     res = minimize(get_neg_posterior, np.zeros(1), args=(y, theta), method='BFGS')
#     mean_lap = np.squeeze(res.x)
#     cov_matrix = np.squeeze(get_fisher_information(res.x, y, theta))
#     var_lap = np.sqrt(1./cov_matrix)
#     return norm(loc = mean_lap, scale = var_lap)