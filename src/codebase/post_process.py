import numpy as np
import pandas as pd
import operator
from numpy.linalg import eigh
from scipy.stats import kde


def get_topn(L, topn):
    """
    Return the top n values a list L, along with their indices
    """
    index = np.empty(topn, dtype=int)
    max_evals = np.empty(topn)
    L_c = L.copy()
    for i in range(topn):
        # find highest element of L_c, with its index
        ind, value = max(enumerate(L_c), key=operator.itemgetter(1))
        index[i] = ind
        max_evals[i] = value
        # delete element to repeat process
        L_c = np.delete(L_c, ind)
    return index, max_evals


def get_topn_eig(M, topn):
    eset = eigh(M)
    L = eset[0]
    P = eset[1]
    index, max_evals = get_topn(L, topn)

    out = dict()
    out["index"] = index
    out["P"] = P[:, index]
    for i in range(topn):
        if out["P"][0, i] < 0:
            out["P"][:, i] = -out["P"][:, i]

    out["L"] = L[index]

    return out


def get_non_zeros(x, prc_min=10, prc_max=90):
    """
    Returns the index of the elements that do not contain
    zero in their quantile interval from prc_min to prc_max.
    Indices are returned in two arrays a1, a2. The i-th non
    zero element of the matrix x is at position [a1[i], a2[i]].
    """
    rcs = np.percentile(x, [prc_min, prc_max], axis=0)
    min_b = rcs[0, :, :]
    max_b = rcs[1, :, :]

    zeros = np.zeros((x.shape[1], x.shape[2]))

    indx = (min_b < zeros) & (zeros < max_b)
    # data['u'][~indx]
    return np.nonzero(~indx)


def kde_mode(x):
    """
    Find the mode of the kernel density estimated by the data.

    :param x: [n,k] array of sample size n each of dimension k
    :return [k,] array of modes for each dimension of x
    """
    nparam_density = kde.gaussian_kde(x)
    return x[np.argsort(nparam_density)[-1]]


def return_post_df(ps, param_name, cn, row, col=None):
    post_df = pd.DataFrame(data=ps[param_name][:, cn, row, col], columns=["val"])
    post_df = post_df.reset_index()
    post_df["param_name"] = param_name
    post_df["cn"] = cn
    post_df["row"] = row
    if col is not None:
        post_df["col"] = col
    return post_df


def samples_to_df(ps, param_name):
    num_chains = ps[param_name].shape[1]
    num_rows = ps[param_name].shape[2]
    if ps[param_name].ndim > 3:
        plot_dims = 2
        num_cols = ps[param_name].shape[3]
    else:
        plot_dims = 1

    post_dfs = []

    for cn in range(num_chains):
        for row in range(num_rows):
            if plot_dims == 2:
                for col in range(num_cols):
                    post_dfs.append(return_post_df(ps, param_name, cn, row, col))
            else:
                post_dfs.append(return_post_df(ps, param_name, cn, row, None))

    return pd.concat(post_dfs, axis=0)


def make_betas_positive(ps, mcmc_samples, num_chains, num_groups):
    for i in range(mcmc_samples):
        for cn in range(num_chains):
            for j in range(num_groups):
                sign1 = np.sign(ps["beta"][i, cn, j, 0, 0])
                sign2 = np.sign(ps["beta"][i, cn, j, 2, 1])
                ps["beta"][i, cn, j, :2, 0] = ps["beta"][i, cn, j, :2, 0] * sign1
                ps["beta"][i, cn, j, 2:, 1] = ps["beta"][i, cn, j, 2:, 1] * sign2
                if "Phi_cov" in ps.keys():
                    ps["Phi_cov"][i, cn, j, 0, 1] = (
                        sign1 * sign2 * ps["Phi_cov"][i, cn, j, 0, 1]
                    )
                    ps["Phi_cov"][i, cn, j, 1, 0] = ps["Phi_cov"][i, cn, j, 0, 1]
    return ps


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
    num_chains = samples.shape[1]
    samples = remove_cn_dimension(samples)
    if samples.ndim > 2:
        rows = samples.shape[1]
        df = form_df(samples, rows)
    else:
        rows = 1
        df = form_df(samples, 1)
    return df

def remove_cn_dimension(samples):
    num_chains = samples.shape[1]
    return np.squeeze(
        np.vstack(
            np.split(samples, num_chains, axis=1)
            ),
            axis=1
    )