import os
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, norm, bernoulli
from numpy.linalg import inv,cholesky
from numpy.random import uniform
from scipy.special import expit, logit


def gen_data(nsim_data = 100, random_seed=None):
    """
    Generate simulated data using means and standard deviations
    from the original data
    """

    if random_seed is not None:
        np.random.seed(random_seed)

    K = 1
    data=dict()
    data['mu'] = -1
    data['Sigma'] = 1.5**2
    nsim_data = 100
    data['y_original'] = norm.rvs(loc=data['mu'], scale=data['Sigma']**(.5), size=nsim_data).reshape(nsim_data,K)
    data['y'] = data['y_original'].copy()
    data['K'] = 1
    data['N'] = nsim_data
    return data


def gen_cov_matrix(dim, scale = 1., random_seed = None):
    """
    Return covariance matrix with values scaled according
    to the input scale.
    Inputs
    ============
    - dim
    - scale

    Output
    ============
    - np. array of shape (dim, dim)
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    A = np.tril(uniform(-scale,scale,size = (dim,dim)))
    C = A @ A.T
    return C


def flatten_corr_matrix_samples(Rs, offset = 0, colnames=None):
    """
    Flatten a [N, K, K ] array of correlation
    matrix samples to a [N,M] array where
    M is the number of of elements below the
    diagonal for a K by K matrix.

    For each sample correlation matrix we care only
    for these M parameters
    Inputs
    ============
    - Rs : samples to flattent out, should be
        of dimension [N,K,K]
    - offset : set 1 to exclude diagonal
    - colnames
    Output
    ============
    -  a dataframe of size [N,M]
    """
    N,K = Rs.shape[0], Rs.shape[1]
    if colnames is None:
        colnames = [str(x) for x in range(K)]

    assert len(colnames) == K, 'colnames should as long as the columns of R'
    cnames = corr_headers(colnames, offset = offset)

    M = len(cnames)
    fRs = np.empty((N,M))
    for i in range (N):
        fRs[i,:] = flatten_corr(Rs[i,:,:], offset = offset)

    fRs = pd.DataFrame(fRs)
    fRs.columns=cnames


    return fRs

def corr_headers(colnames, offset = 0):
    """
    Return the headers for the DataFrame
    of the correlation coefficients.

    If the original correlation matrix is
    of size [K, K ], then corr_headers
    takes in K headers and returns an
    array of size [M,], where
    M is the number of of elements above the
    diagonal for a K by K matrix.

    To be used together with `flatten_corr`.
    Start with a correlation matrix
    R, get the correlation coefficients with
    flatten_corr(R) and turn it into a
    dataframe with columns headers
    corr_headers(colnames) where
    colnames are the headers of the data.

    Inputs
    ============
    - R : headers of the data.
    Output
    ============
    -  an array of size [M,]
    """
    colnames = np.array(colnames)
    dfcols = list(zip(colnames[np.triu_indices(colnames.shape[0], k=offset)[0]],\
                 colnames[np.triu_indices(colnames.shape[0], k=offset)[1]]))
    return dfcols


def flatten_corr(a, offset = 0):
    """
    Flatten a [K, K ] correlation
    matrix to [M,] array where
    M is the number of of elements above the
    diagonal for a K by K matrix.

    Inputs
    ============
    - R : matrix to flattent out, should be
        of dimension [K,K]
    Output
    ============
    -  an array of size [M,]
    """
    return a[np.triu_indices(a.shape[0], k=offset)]


def C_to_R(M):
    """
    Send a covariance matrix M to the corresponding
    correlation matrix R
    Inputs
    ============
    - M : covariance matrix
    Output
    ============
    - correlation matrix
    """
    d = np.asarray(M.diagonal())
    d2 = np.diag(d**(-.5))
    R = d2 @ M @ d2
    return R


def thin(x, rate = 10):
    """
    Thin an array of numbers by choosing every
    nth sample
    """
    return x[::rate]


def corr_columnwise(X,Y):
    """
    X = [x1,x2,..xN]
    Y = [y1,y2,..yN]
    where xi,yi are d-dim vectors.
    outputs corr which is a d-dim vector
    where corr[i] = corrcoef(xi,yi)
    """

    d = X.shape[1]
    assert X.shape == Y.shape

    out = np.empty(d)
    for i in range(d):
        out[i] = np.corrcoef(X[:,i], Y[:,i], rowvar = False)[0,1]

    return out


def oned_f_logit_inv(x):
    """
    (-inf,inf) -> [-1,1]
    """
    return (np.exp(x)-1)/(np.exp(x)+1)

f_logit_inv = np.vectorize(oned_f_logit_inv)

def oned_f_logit(x):
    """
    from [-1,1] -> (-inf,inf)
    """
    return np.log(1+x) - np.log(1-x)

f_logit = np.vectorize(oned_f_logit)


def theta_transform(theta):
    """
    parameter space -> constrained (sampling) space
    """
    oned=False
    try:
        N,k = theta.shape
    except:
        k = theta.shape[0]
        N =1
        oned=True

    assert k==4

    theta = theta.astype(float)
    theta_trans = theta.reshape((N,k)).copy()
    theta_trans[:,2] = np.log(theta_trans[:,2])
    theta_trans[:,3] = f_logit(theta_trans[:,3])
    if oned==True:
        theta_trans = theta_trans[0]
    return theta_trans


def theta_transform_inv(theta, k=4):
    """
    constrained (sampling) space -> parameter space
    """
    assert k==4

    oned=False
    try:
        N,k = theta.shape
    except:
        k = theta.shape[0]
        N =1
        oned=True
    theta = theta.astype(float)

    theta_trans = theta.reshape((N,k)).copy()
    theta_trans[:,2] = np.exp(theta_trans[:,2])
    theta_trans[:,3] = f_logit_inv(theta_trans[:,3])

    if oned==True:
        theta_trans = theta_trans[0]
    return theta_trans


def return_params(theta):
    """
    parameter vector -> dict(mu, Sigma)
    """
    oned = False
    try:
        N,k = theta.shape
    except:
        k = theta.shape[0]
        N =1
        oned = True

    theta_trans = theta.reshape((N,k))

    params = dict()
    params['mu'] = theta_trans[:,:2]

    Sigmas = np.empty((N,2,2))
    for i in range(N):
        d = np.ones(2)
        d[0] = theta_trans[i,2]**.5
        D = np.diag(d)
        R = np.eye(2)
        R[0,1] = theta_trans[i,3]
        R[1,0] = R[0,1]
        Sigmas[i] = D @ R @ D

    params['Sigma'] = Sigmas

    if oned == True:
        params['mu'] = params['mu'][0]
        params['Sigma'] = params['Sigma'][0]

    return params


def check_posdef(R):
    """
    Check that matrix is positive definite
    Inputs
    ============
    - matrix
    """
    try:
        cholesky(R)
    except NameError:
        print("\n Error: could not compute cholesky factor of R")
        raise
    return 0    
