import numpy as np
from scipy.stats import multivariate_normal, bernoulli
from tqdm import tqdm
from scipy.special import expit, logsumexp
from pdb import set_trace
import pandas as pd

def to_str_pattern(y0):
    if np.ndim(y0) == 1:
        return "".join(y0.astype(str))
    if np.ndim(y0) == 2:
        y = pd.DataFrame(y0)
        yresp = y.apply(lambda x: "".join(x.astype(str)), axis=1)
        return yresp


def to_nparray_data(yresp):
    if type(yresp) == str:
        return np.array(list(yresp)).astype(int)
    else:
        J = len(yresp[0])
        N = yresp.shape[0]
        res = np.empty((N, J))
        for i in range(N):
            res[i] = np.array(list(yresp[i])).astype(int)
        return res


def get_factor_model_posteriors(ps, m, g):
    dim_K = ps["beta_b"].shape[-1]
    m_alpha = ps["alpha_b"][m, g]
    if "Marg_cov_bin" in ps.keys():
        m_Marg_cov = ps["Marg_cov_bin"][m, g]  # covariance matrix is pooled
        post_y_sample = multivariate_normal.rvs(
            mean=m_alpha, cov=m_Marg_cov, size=1
        )
    else:
        m_beta = ps["beta_b"][m, g]  # covariance matrix is pooled
        if "Phi_cov" in ps.keys():
            m_Phi_cov = ps["Phi_cov"][m, g]  # covariance matrix is pooled
        else:
            m_Phi_cov = np.eye(dim_K)
        zz_from_prior = multivariate_normal.rvs(
            mean=np.zeros(dim_K), cov=m_Phi_cov, size=1
        )
        post_y_sample = m_alpha + zz_from_prior @ m_beta.T
    return post_y_sample


def get_saturated_model_posteriors(ps, m, g):
    m_alpha = ps["alpha_b"][m, g]
    m_Marg_cov = ps["Marg_cov_bin"][m, g]  # covariance matrix is pooled
    post_y_sample = multivariate_normal.rvs(mean=m_alpha, cov=m_Marg_cov, size=1)
    return post_y_sample


def get_combined_logscore(ps, data, nsim, model_num):
    mcmc_length = ps["alpha_b"].shape[0]
    if nsim > mcmc_length:
        print("nsim > posterior sample size")
        print("Using nsim = %d" % mcmc_length)
        nsim = mcmc_length
    skip_step = int(mcmc_length / nsim)
    
    
    num_of_groups = len(np.unique(data["grp"]))
    logscores = np.empty(num_of_groups, dtype=float)

    for g in range(num_of_groups):
        y_for_group_g = data["yc"][data["grp"] == (g + 1)]

        data_ptrn_for_group_g = to_str_pattern(
            data["yb"][
                data["grp"] == (g + 1)
            ]
        )
        test_size = y_for_group_g.shape[0]
        y_lklhds = np.empty(test_size, dtype = float)
        loglik_theta_m = np.empty(nsim, dtype=float)
        for i in tqdm(range(test_size)):
            for m_ind in range(nsim):
                m = m_ind * skip_step
                mean =  ps['alpha_c'][m, g]
                Cov = ps['Marg_cov_cont'][m, g]
                logscore_cont = multivariate_normal.logpdf(
                    y_for_group_g[i],
                    mean=mean,
                    cov = Cov
                    )
                if model_num in [1, 2, 4, 5]:
                    post_y = get_factor_model_posteriors(ps, m, g)
                elif model_num in [0, 3]:
                    post_y = get_saturated_model_posteriors(ps, m, g)


                logscore_bin = bernoulli.logpmf(
                    k=to_nparray_data(data_ptrn_for_group_g[i]),
                    p=expit(post_y)
                ).sum()
                loglik_theta_m[m_ind] = logscore_cont + logscore_bin

            #  log of average likelihood per person
            y_lklhds[i] = logsumexp(loglik_theta_m) - np.log(nsim)
        logscores[g] = y_lklhds.sum() - np.log(test_size)
    return -logscores.sum()