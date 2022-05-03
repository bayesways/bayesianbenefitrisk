from codebase.classes_smclvm import PariclesSMCLVM
from codebase.ibis import essl, corrcoef_2D
from codebase.ibis import run_stan_model
from codebase.post_process import remove_cn_dimension
from codebase.mcmc_tlk_latent import gen_latent_weights_master
import numpy as np
from tqdm import tqdm
from codebase.file_utils import (
    save_obj,
    load_obj,
)
from copy import deepcopy
from scipy.special import logsumexp
from pdb import set_trace


def get_init_mcmc(data, param_names, size):
    sm = load_obj("sm", "log/compiled_models/sequential/model1/")
    fit_run = run_stan_model(
        data=data,
        compiled_model=sm,
        num_samples=int(size*10), 
        num_warmup=1000,
        num_chains=1,  # don't change
        adapt_engaged=True,
    )
    init_particles = fit_run.extract(permuted=False, pars=param_names)

    for name in param_names:
        init_particles[name] = remove_cn_dimension(init_particles[name])[::10]

    return init_particles
    

def run_smclvm(
    exp_data,
    init_t,
    size,
    gen_model,
    log_dir,
    degeneracy_limit=0.5,
    name="ibis_lvm",
    run_init_mcmc=False
):

    param_names = [
        "alpha",
        "alpha_c",
        "alpha_b",
        "beta_b",
        "beta_c",
        'beta1',
        "theta",
        "Marg_cov_cont",
    ]
    stan_input_names = ["alpha", 'beta1', 'beta_b', 'theta', 'z']
    stan_output_names = param_names + ["z", 'y']
    latent_names = ["zz", 'yy']
    jitter_corrs = dict()
    for t in range(exp_data.size-init_t):
        jitter_corrs[t] = dict()
    particles = PariclesSMCLVM(
        name="seq_fa",
        size=size,
        param_names=param_names,
        stan_input_names = stan_input_names,
        stan_output_names= stan_output_names,
        latent_names=latent_names,
        hmc_adapt_nsim=400,
        hmc_post_adapt_nsim=10,
    )
    particles.set_log_dir(log_dir)
    if gen_model:
        particles.compile_prior_model()
        particles.compile_model()
    else:
        particles.load_prior_model()
        particles.load_model()

    if run_init_mcmc:
        mcmc_samples_initial = get_init_mcmc(
            exp_data.get_stan_data_upto_t(init_t),
            param_names,
            particles.size
            )
        particles.initialize_particles(mcmc_samples_initial)
        exp_data.keep_data_after_t(init_t)
    else:
        particles.sample_prior_particles(
            exp_data.get_stan_data_at_t2(0)
        )  # sample prior particles

    log_lklhds = np.empty(exp_data.size)
    degeneracy_limit = 0.5
    particles.initialize_latentvars(exp_data.get_stan_data())
    particles.reset_weights()  # set weights to 0
    particles.initialize_counter(exp_data.get_stan_data())
    particles.initialize_scores(exp_data.get_stan_data())

    for t in tqdm(range(exp_data.size)):
        particles.sample_latent_variables(exp_data.get_stan_data_at_t(t), t)
        particles.get_theta_incremental_weights(exp_data.get_stan_data_at_t(t), t)
        log_lklhds[t] = particles.get_loglikelihood_estimate()
        particles.update_weights()

        if (essl(particles.weights) < degeneracy_limit * particles.size) and (
            t + 1
        ) < exp_data.size:
            particles.add_ess(t)
            particles.resample_particles()
            particles.gather_variables_prejitter(
                t + 1, exp_data.get_stan_data_upto_t(t + 1)
            )

            ## add corr of param before jitter
            pre_jitter = dict()
            for p in stan_output_names:
                pre_jitter[p] = np.copy(particles.particles[p])
            ###
            particles.jitter(exp_data.get_stan_data_upto_t(t + 1), t + 1)

            particles.gather_variables_postjitter(
                t + 1, exp_data.get_stan_data_upto_t(t + 1)
            )

            # add corr of param
            # for p in param_names:
            #     if p not in ["zz", "yy"]:
            #         jitter_corrs[t][p] = corrcoef_2D(
            #             pre_jitter[p], particles.particles[p]
            #         )
            ###
            if t>40:
                particles.check_particles_are_distinct()
            particles.reset_weights()
            particles.get_population_scores_at_t(t, False)
            
        else:
            particles.get_population_scores_at_t(t, True)


        save_obj(t, "t", log_dir)
        save_obj(particles, "particles", log_dir)
        save_obj(jitter_corrs, "jitter_corrs", log_dir)
        save_obj(log_lklhds, "log_lklhds", log_dir)

    print("\n\n")
    marg_lklhd = np.exp(logsumexp(log_lklhds))
    print("Marginal Likelihood %.5f" % marg_lklhd)
    save_obj(marg_lklhd, "marg_lklhd", log_dir)

    output = dict()
    output["particles"] = particles
    output["log_lklhds"] = log_lklhds
    output["marg_lklhd"] = marg_lklhd
    output["jitter_corrs"] = jitter_corrs
    return output
