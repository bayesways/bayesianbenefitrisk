import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, norm
from tqdm import tqdm
from numpy.linalg import det, inv
from codebase.file_utils import save_obj, load_obj
from scipy.special import expit
from pdb import set_trace


def ff2(yy, model_Sigma, p):
    sample_S = np.cov(yy, rowvar=False)
    ldS = np.log(det(sample_S))
    iSigma = inv(model_Sigma)
    ldSigma = np.log(det(model_Sigma))
    n_data = yy.shape[0]
    ff2 = (n_data - 1) * (ldSigma + np.sum(np.diag(sample_S @ iSigma)) - ldS - p)
    return ff2


def compute_D(data, ps, mcmc_iter, pred=True):
    J = ps["alpha_c"][mcmc_iter, 0].shape[0]
    if pred == True:
        n = data["yc"].shape[0]
        y_pred = np.empty((n, J))
        for i in range(n):
            y_pred[i, :] = multivariate_normal.rvs(
                mean=ps["alpha_c"][mcmc_iter, i], cov=ps["Marg_cov_cont"][mcmc_iter, i]
            )
        grp_ff2_scores = np.empty(3)
        for i in range(3):
            grp_ff2_scores[i] = ff2(
                y_pred[data["grp"] == (i + 1)], ps["Marg_cov_cont"][mcmc_iter, i], p=J
            )
        return sum(grp_ff2_scores)
    else:
        grp_ff2_scores = np.empty(3)
        for i in range(3):
            grp_ff2_scores[i] = ff2(
                data["yc"][data["grp"] == (i + 1)],
                ps["Marg_cov_cont"][mcmc_iter, i],
                p=J,
            )
        return sum(grp_ff2_scores)


def get_PPP(data, ps, nsim):
    mcmc_length = ps["alpha"].shape[0]
    skip_step = int(mcmc_length / nsim)
    PPP_vals = np.empty((nsim, 2))
    for m_ind in tqdm(range(nsim)):
        m = skip_step * m_ind
        PPP_vals[m_ind, 0] = compute_D(data, ps, m, pred=False)
        PPP_vals[m_ind, 1] = compute_D(data, ps, m, pred=True)
    return PPP_vals

