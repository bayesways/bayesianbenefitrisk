import numpy as np
from codebase.ibis import (
    compile_model,
    sample_prior_particles,
    get_resample_index,
    run_mcmc,
    get_initial_values_dict,
    exp_and_normalise,
    remove_chain_dim,
)
from codebase.ibis_tlk import gen_weights_master
from codebase.file_utils import save_obj, load_obj, make_folder, path_backslash
from codebase.resampling_routines import multinomial
from scipy.special import logsumexp
from shutil import copyfile
from pdb import set_trace


class Particles:
    def __init__(
        self,
        name,
        size,
        param_names,
        stan_input_names,
        stan_output_names,
        latent_names,
        hmc_adapt_nsim,
        hmc_post_adapt_nsim,
    ):
        self.name = name
        self.size = size
        self.param_names = param_names
        self.stan_input_names = stan_input_names
        self.stan_output_names = stan_output_names
        self.latent_names = latent_names
        self.hmc_adapt_nsim = hmc_adapt_nsim
        self.hmc_post_adapt_nsim = hmc_post_adapt_nsim

    def set_log_dir(self, log_dir):
        self.log_dir = log_dir

    def load_prior_model(self):
        self.compiled_prior_model = load_obj(
            "sm_prior", "./log/compiled_models/sequential/model1/" 
        )
        copyfile(
            "./log/compiled_models/sequential/model1/model_prior.txt" ,
            "%s/model_prior.txt" % self.log_dir,
        )

    def load_model(self):
        self.compiled_model = load_obj(
            "sm", "./log/compiled_models/sequential/model1/"
        )
        copyfile(
            "./log/compiled_models/sequential/model1/model.txt",
            "%s/model.txt" % self.log_dir,
        )
        self.mass_matrix = None
        self.stepsize = None

    def compile_prior_model(self):
        self.compiled_prior_model = compile_model(
            prior=True, log_dir=self.log_dir, save=True
        )
        copyfile(
            "./log/compiled_models/sequential/model1/model_prior.txt",
            "%s/model_prior.txt" % self.log_dir,
        )

    def compile_model(self):
        self.compiled_model = compile_model(
            prior=False, log_dir=self.log_dir, save=True
        )
        copyfile(
            "./log/compiled_models/sequential/model1/model.txt",
            "%s/model.txt" % self.log_dir,
        )
        self.mass_matrix = None
        self.stepsize = None

    def sample_prior_particles(self, data, names=None):
        if names is None:
            names = self.param_names
        self.particles = sample_prior_particles(
            data=data,
            sm_prior=self.compiled_prior_model,
            param_names=names,
            num_samples=self.size,
            num_chains=1,
            log_dir=self.log_dir,
        )
    
    def initialize_particles(self, init_particles):
        self.particles = init_particles

    def reset_weights(self):
        self.weights = np.zeros(self.size)

    def get_incremental_weights(self, data):
        self.incremental_weights = gen_weights_master(
            data, self.particles, self.size
        )

    def update_weights(self):
        weights = self.weights.copy()
        weights = weights + self.incremental_weights
        self.weights = weights.copy()

    def resample_particles(self):
        resample_index = get_resample_index(self.weights, self.size)
        for name in self.param_names:
            samples = self.particles[name][resample_index].copy()
            self.particles[name] = samples

    def get_particles_at_position_m(self, m):
        values_dict = dict()
        for name in self.param_names:
            values_dict[name] = self.particles[name][m]
        return values_dict

    def jitter_and_save_mcmc_parms(self, data, m=0):
        fit_run = run_mcmc(
            data=data,
            sm=self.compiled_model,
            num_samples=self.hmc_post_adapt_nsim,  # normally 20
            num_warmup=self.hmc_adapt_nsim,  # normally 1000
            num_chains=1,  # don't change
            log_dir=self.log_dir,
            initial_values=self.get_particles_at_position_m(m),
            load_inv_metric=False,
            adapt_engaged=True,
        )
        self.mass_matrix = fit_run.get_inv_metric(as_dict=True)
        self.stepsize = fit_run.get_stepsize()
        last_position = fit_run.get_last_position()[0]  # select chain 1
        for name in self.param_names:
            self.particles[name][m] = last_position[name]

    def jitter_with_used_mcmc_params(self, data, m):
        fit_run = run_mcmc(
            data=data,
            sm=self.compiled_model,
            num_samples=self.hmc_adapt_nsim,  # normally 20
            num_warmup=0,
            num_chains=1,  # don't change
            log_dir=self.log_dir,
            initial_values=self.get_particles_at_position_m(m),
            inv_metric=self.mass_matrix,
            adapt_engaged=False,
            stepsize=self.stepsize,
        )
        last_position = fit_run.get_last_position()[0]  # select chain 1
        for name in self.param_names:
            self.particles[name][m] = last_position[name]

    def jitter(self, data):
        self.jitter_and_save_mcmc_parms(data, 0)
        for m in range(1, self.size):
            self.jitter_with_used_mcmc_params(data, m)

    def get_loglikelihood_estimate(self):
        return logsumexp(self.incremental_weights + self.weights) - logsumexp(
            self.weights
        )
