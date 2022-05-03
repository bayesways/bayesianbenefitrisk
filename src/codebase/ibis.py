import pystan
import sys, os
from shutil import copyfile
import argparse
import numpy as np
from codebase.file_utils import save_obj, load_obj, make_folder, path_backslash
from codebase.resampling_routines import multinomial
from scipy.stats import bernoulli, multivariate_normal
from scipy.special import expit, logsumexp
from tqdm import tqdm
from pdb import set_trace


def compile_model(prior, log_dir, save=True):

    model_bank_path = "./log/compiled_models/sequential/model1/"

    if prior: 
        model_path = 'codebase/stan_code/sequential/ibis/model_prior.stan'
    else:
        model_path ='codebase/stan_code/sequential/ibis/model.stan'
    with open(model_path, "r") as file:
        model_code = file.read()

    sm = pystan.StanModel(model_code=model_code, verbose=False)

    if save:
        if prior:
            save_obj(sm, "sm_prior", log_dir)
            save_obj(sm, "sm_prior", model_bank_path)
            file = open("%smodel_prior.txt" % model_bank_path, "w")
            file.write(model_code)
            file.close()
            copyfile(
                "%smodel_prior.txt" %model_bank_path,
                "%s/model_prior.txt"%log_dir
                )
            
        else:
            save_obj(sm, "sm", log_dir)
            save_obj(sm, "sm", model_bank_path)
            file = open("%smodel.txt" % model_bank_path, "w")
            file.write(model_code)
            file.close()
            copyfile(
                "%smodel.txt" %model_bank_path,
                "%s/model.txt"%log_dir
                )
    return sm


def sample_prior_particles(
    data, sm_prior, param_names, num_samples, num_chains, log_dir
):
    fit_run = sm_prior.sampling(
        data=data,
        iter=num_samples,
        warmup=0,
        chains=num_chains,
        algorithm="Fixed_param",
        n_jobs=1
    )
    particles = remove_chain_dim(
        fit_run.extract(permuted=False, pars=param_names), param_names, num_samples
    )
    return particles


def remove_chain_dim(ps, param_names, num_samples):
    if num_samples > 1:
        for name in param_names:
            ps[name] = np.copy(ps[name][:, 0])
    else:
        for name in param_names:
            ps[name] = np.copy(ps[name][0, 0])
    return ps


def get_initial_values_dict(particles, m):
    particles_dict = dict()
    for n in particles["param_names"]:
        particles_dict[n] = particles[n][m, 0]
    return particles_dict


def set_last_position(particles, m, last_position):
    for n in particles["param_names"]:
        particles[n][m] = last_position[n]
    return particles


init_values = dict()


def set_initial_values(params):
    global init_values  # Needed to modify global copy of globvar
    init_values = params


def initf1():
    return init_values


def run_stan_model(
    data,
    compiled_model,
    num_samples,
    num_warmup,
    num_chains,
    initial_values=None,
    inv_metric=None,
    adapt_engaged=False,
    stepsize=None,
):

    if initial_values is not None:
        set_initial_values(initial_values)

    control = {
        "metric": "diag_e",  # diag_e/dense_e
        "adapt_delta": 0.99,
        "max_treedepth": 14,
        "adapt_engaged": adapt_engaged,
    }

    if inv_metric is not None:
        control["inv_metric"] = inv_metric
    if stepsize is not None:
        control["stepsize"] = stepsize
    
    fit_run = compiled_model.sampling(
        data=data,
        iter=num_samples + num_warmup,
        warmup=num_warmup,
        chains=num_chains,
        init=initf1,
        control=control,
        n_jobs=1,
        check_hmc_diagnostics=False,
    )

    return fit_run


def run_mcmc(
    data,
    sm,
    num_samples,
    num_warmup,
    num_chains,
    log_dir,
    initial_values=None,
    inv_metric=None,
    load_inv_metric=False,
    save_inv_metric=False,
    adapt_engaged=False,
    stepsize=None,
):

    if load_inv_metric:
        inv_metric = load_obj("inv_metric", log_dir)

    fit_run = run_stan_model(
        data,
        compiled_model=sm,
        num_samples=num_samples,
        num_warmup=num_warmup,
        num_chains=num_chains,
        initial_values=initial_values,
        inv_metric=inv_metric,
        adapt_engaged=adapt_engaged,
        stepsize=stepsize,
    )

    if save_inv_metric:
        inv_metric = fit_run.get_inv_metric(as_dict=True)
        save_obj(inv_metric, "inv_metric", log_dir)

    return fit_run


def exp_and_normalise(lw):
    """Exponentiate, then normalise (so that sum equals one).
    Arguments
    ---------
    lw: ndarray
        log weights.
    Returns
    -------
    W: ndarray of the same shape as lw
        W = exp(lw) / sum(exp(lw))
    Note
    ----
    uses the log_sum_exp trick to avoid overflow (i.e. subtract the max
    before exponentiating)
    See also
    --------
    log_sum_exp
    log_mean_exp
    """
    w = np.exp(lw - lw.max())
    return w / w.sum()


def essl(lw):
    """ESS (Effective sample size) computed from log-weights.
    Parameters
    ----------
    lw: (N,) ndarray
        log-weights
    Returns
    -------
    float
        the ESS of weights w = exp(lw), i.e. the quantity
        sum(w**2) / (sum(w))**2
    Note
    ----
    The ESS is a popular criterion to determine how *uneven* are the weights.
    Its value is in the range [1, N], it equals N when weights are constant,
    and 1 if all weights but one are zero.
    """
    w = np.exp(lw - lw.max())
    return (w.sum()) ** 2 / np.sum(w ** 2)


def get_resample_index(weights, size):
    nw = exp_and_normalise(weights)
    np.testing.assert_allclose(1.0, nw.sum())
    return multinomial(nw, size)

def post_process_sign(ps):
    nsim = ps["beta"].shape[0]
    for n in range(nsim):
        sign = np.sign(ps["beta"][n, 0])
        ps["beta"][n] = (
            sign
            * ps["beta"][
                n,
            ]
        )
    return ps

def corrcoef_2D(a, b):
    assert a.shape == b.shape
    corrs = np.empty(a.shape[1:])
    if corrs.ndim == 1:
        n = corrs.shape[0]
        for i in range(n):
            corrs[i] = np.corrcoef(a[:,i], b[:,i])[0,1]
    else:
        n = corrs.shape[0]
        p = corrs.shape[1]
        for i in range(n):
            for j in range(p):
                corrs[i,j] = np.corrcoef(a[:,i,j], b[:,i,j])[0,1]
    return corrs
