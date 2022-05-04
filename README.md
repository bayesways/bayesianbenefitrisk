# MCDA Application of Bayesian SEM

## Setup

Initial install:

    $ conda env create -f environment.yml
    $ source activate benefitriskanalysis

Update:

    $ conda env update -f environment.yml
    
Export: 

    $ conda env export > environment.yml

Deactivate Environment:

    $ source deactivate

Remove:

    $ conda remove -n benefitriskanalysis --all


Convert `a.ipynb` to `a.py`:

    $ jupyter nbconvert --to script a.ipynb

## Data 

This repository contains strictly synthetic data to facilitate demonstrating the methodology developed in the paper. To create the synthetic dataset and save it in the `./dat/` folder run the following

    $ cd src
    $ python create_synthetic_data.py

Any reference to "real data" in this repository refers to the synthetic data created in this step.


## How to use:

1. Run experiment on data with `run_clinical_trial.py` to fit the model. Pass the flag `-cv cv` to fit models for cross validation. The results will be saved in `./src/log`. You can select the model from the available options (see list of available models further down this page).
2. Compute PPP values and Cross Validation Indexes with `compute_ppp.py` and `compute_cv.py` respectively passing the directory where the results are stored in (from step 1). 
3. Run simulations for control and treatment groups with `run_simulations.py`. This will create datasets and fit the models all in one script. Works similarly to `run_clinical_trial.py`.
4. Run sequential algorithm used in the paper with `run_seq_ct.py`. This script takes an argument for `-sim_case 0` loads the synthetic data in `./dat`, and `-sim_case 1` generates new synthetic data. The scripts also accepts `-run_init_mcmc 1` as an argument to run an MCMC chain to initialize the particles using the first `t` (`-init_t t`) data points. 
5. Read the outputs of the run with the notebook `MCDA Results.ipynb`.


Additional Notes on Usage: 

* Simulations accept seeds arguments to be used for the generation of data
* We chose the best model for the real clinical trial data based on PPP and CV. For that model, and only for that model, we created a sequential algorithm which can be run with `run_seq_ct.py`. 
* the final results depend on the MCDA weights which are hardcoded. 
* Simulation scenarios include difference in group underlying means and correlations among observed variables: `args.diff` controls the scenario of the group difference, `args.corr_sim` controls the scenario of correlation among variables. For `corr_sim=1` the user can control additionally the correlation among the factors `args.rho`. The other scenario `corr_sim=2`  uses hard coded correlation values. 
* Each subject is given a group number based on the arm they belong as `AVM: 1`, `MET: 2`, `RSG: 3`


## Models

We describe the models in <short name>, <real_data_fit.py number>, <stan_model_name> and a short description if needed

* S, num 0, `saturated.stan`, saturated model
* S-I, num 3, `saturated__2.stan`, satutrated model with identity correlation 
* EZ1 model, num 1, `factor_model_1.stan`, EZ model with independent factors  
* EZ2 model, num 4, `factor_model_2.stan`, EZ model with correlated factors 
* AZ1 model, num 5, `factor_model_4.stan`, AZ model with correlated correlated errors for binary only
* AZ2 model, num 2, `factor_model_3.stan`, AZ model with cross loadings and correlated errors for binary only

Pooled models differ in that the covariance matrix is the same among all groups.
Any of the above models can be turned to a pooled model by specifying `cov_num=1` as a data input.

## Useful tips

On local system, to save storage space you can remove old pickled files with

    $ find . type f -name '*.p' -delete
