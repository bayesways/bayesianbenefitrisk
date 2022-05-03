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

This repository contains strictly synthetic data to facilitate demonstrating the usages of the code. To create the synthetic dataset and save it in the `./dat/` folder run the following

    $ cd src
    $ python create_synthetic_data.py


## How to use:

1. Run experiment on data with `run_clinical_trial.py`
2. Compute PPP values and Cross Validation Indexes with `compute_ppp.py` and `compute_cv.py` respectively. 
3. Run simulations on synthetic clinical data with `run_simulations.py`. 
4. Run sequential algorithm with `run_seq_ct.py`. This script takes an argument for `sim_case=0` loads the synthetic data in `./dat`, and `sim_case=1` generates new synthetic data. The scripts also takes an argument for running the first `t` data points with the MCMC method to initialiaze the particles. To run without the MCMC initialization run choose `run_init_mcmc=False`.
5. Sequential populations scores are added to the IBIS algorithm
6. MCDA results offline and online algorithms are in notebooks `4.1`, `4.2` and `4.3`


Additional Notes on Usage: 

* Simulations accept seeds arguments to be used for the generation of data
* We chose the best model for the real clinical trial data based on PPP and CV. For that model, and only for that model, we created a sequential algorithm which can be run with `run_seq_ct.py`. 
* The sequential algorithm results can be checked against the full MCMC results with notebook `2. Simulation Results` and `3. Real Data Results`
* the final results depend on the MCDA weights which are hardcoded. 
* Simulation scenarios include difference in group underlying means and correlations among observed variables: `args.diff` controls the scenario of the group difference, `args.corr_sim` controls the scenario of correlation among variables. For `corr_sim=1` the user can control additionally the correlation among the factors `args.rho`. The other scenario `corr_sim=2`  uses hard coded correlation values. 
* Each subject is given a group number based on the arm they belong as `AVM: 1`, `MET: 2`, `RSG: 3`


### Models

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