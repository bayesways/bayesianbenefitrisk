import numpy as np
from scipy.stats import bernoulli, multivariate_normal, logistic, norm
from scipy.special import expit, logsumexp
from pdb import set_trace


def gen_latent_weights_master(
    latent_model_num,
    data,
    y_latent,
    bundle_size
    ):
    if latent_model_num == 1:
        return get_latent_weights_1(
            data,
            y_latent,
            bundle_size
            )

def get_latent_weights_1(
    data,
    y_latent,
    bundle_size
    ):
    """
    data in NxJ
    weights are of size NxK 
    where K is the size of each bundle. 

    For each data point D_t we compute
    weights p(D_t | Z_t^k) for k:1...K

    And return the weight of each bundle
    as the average across k 
    """
    weights = np.empty((bundle_size, data['N']))
    if np.ndim(data['D']) == 1 :
        data['D'] = data['D'].reshape((data['N'], data['J']))
        # for m in range(bundle_size):
        #     weights[m,0] = bernoulli.logpmf(
        #         data['D'],
        #         p=expit(y_latent[m])
        #     ).sum()
    # elif np.ndim(data['D']) == 2 :
    for m in range(bundle_size):
        for t in range(data['N']):
            weights[m,t] = bernoulli.logpmf(
                data['D'][t],
                p=expit(y_latent[m,t])
            ).sum()
    # else:
    #     exit
    return weights

def generate_latent_variables(
    data_N,
    data_J,
    data_K,
    alpha,
    beta):
    latent_vars = dict()
    zz = norm.rvs(
        size = data_N*data_K
        ).reshape((data_N,data_K))
    y_latent = np.empty((
        data_N,
        data_J))
    y_latent = np.squeeze(alpha) +\
        zz @ np.squeeze(beta).reshape(data_J, data_K).T
    latent_vars['z'] = zz
    latent_vars["y"] = y_latent
    return latent_vars

def generate_latent_variables_bundle(
    bundle_size,
    data_N,
    data_J,
    data_K,
    alpha,
    beta):
    latent_vars = dict()
    zz = np.empty((
        bundle_size,
        data_N,
        data_K))
    y_latent = np.empty((
        bundle_size,
        data_N,
        data_J))
    for m in range(bundle_size):
        bundle_vars = generate_latent_variables(        
            data_N,
            data_J,
            data_K,
            alpha,
            beta)
        zz[m] = bundle_vars['z']
        y_latent[m] = bundle_vars["y"]
    latent_vars['z'] = zz
    latent_vars["y"] = y_latent
    return latent_vars
