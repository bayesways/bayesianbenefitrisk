import numpy as np
import pandas as pd
import pystan
from codebase.file_utils import save_obj, load_obj
from sklearn.model_selection import KFold
from codebase.data import get_final_dataset
from codebase.tlk import create_group_data
from codebase.mcda import scaled_invlogit_perrow, get_scores_perrow
from scipy.stats import multivariate_normal
import pdb


def run_fit_for_real_data_no_pool(
    log_dir,
    existing_directory,
    stan_model,
    gen_data,
    gen_model,
    data_sim,
    ppp_cv,
    n_splits,
    cv_random_seed,
    print_model,
    num_samples,
    num_warmup,
    num_chains,
    model_path
    ):
    ############################################################
    ################ Create Data or Load ##########

    if existing_directory is None or gen_data:
        
        if data_sim == 0:
            group_data = dict()
            df = pd.read_csv("../dat/synthetic_dataset.csv")
            tmp_dct = dict()
            y = df.iloc[:, 2:].values
            df['atrtgrp'] = df['atrtgrp'].replace({
                'AVM':1,
                'MET':2,
                'RSG':3
            })
            tmp_dct['grp'] = df['atrtgrp'].astype(int)
            tmp_dct['K'] = 6
            tmp_dct['Kc'] = 2
            tmp_dct['Kb'] = 4
            tmp_dct['y'] = y
            tmp_dct['N'] = df.shape[0]
            tmp_dct['number_of_groups'] = 3
            group_data = tmp_dct
        else:
            print("data_sim needs to be in {0,1}")

        if ppp_cv == 'ppp':  # run PPP

            stan_data = dict(
                N=group_data['N'],
                Kb=group_data['Kb'],
                Kc=group_data['Kc'],
                K=group_data['K'],
                yc=group_data['y'][:, :2],
                yb=group_data['y'][:, 2:].astype(int),
                grp = group_data['grp'],
                number_of_groups = 3
                )
            print("\n\nSaving data at %s" % log_dir)
            save_obj(stan_data, 'stan_data', log_dir)
            save_obj(group_data, 'group_data', log_dir)
        elif ppp_cv == 'cv':  # run CV
            stan_data = dict()
            complete_grp_data = dict()
            X = group_data['y']
            Xgrp = group_data['grp']
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=cv_random_seed)
            kf.get_n_splits(X)

            for (fold_index, (train_index, test_index)) in enumerate(
                    kf.split(X)
            ):
                data_fold = dict()
                data_fold['y_train'],\
                    data_fold['y_test'] = X[train_index], X[test_index]
                data_fold['grp_train'],\
                    data_fold['grp_test'] = Xgrp[train_index], Xgrp[test_index],
                data_fold['N_train'], \
                    data_fold['N_test'] = data_fold['y_train'].shape[0],\
                    data_fold['y_test'].shape[0]
                train_data_fold = dict(
                    N=data_fold['N_train'],
                    Kb=group_data['Kb'],
                    Kc=group_data['Kc'],
                    K=group_data['K'],
                    yb=data_fold['y_train'][:, 2:].astype(int),
                    yc=data_fold['y_train'][:, :2],
                    grp=data_fold['grp_train'],
                    number_of_groups = 3

                )
                test_data_fold = dict(
                    N=data_fold['N_test'],
                    Kb=group_data['Kb'],
                    Kc=group_data['Kc'],
                    K=group_data['K'],
                    yb=data_fold['y_test'][:, 2:].astype(int),
                    yc=data_fold['y_test'][:, :2],
                    grp=data_fold['grp_test'],
                    number_of_groups = 3
                )
                complete_grp_data[fold_index] = dict(
                    train=train_data_fold, test=test_data_fold)
            stan_data = complete_grp_data
            save_obj(stan_data, 'stan_data', log_dir)
            save_obj(group_data, 'group_data', log_dir)
            print("\n\nSaving data folds at %s" % log_dir)

        else:
            print("-cv needs to be 'ppp' or 'cv'")

    else:
        print("\n\nReading data from directory %s" % log_dir)
        stan_data = load_obj("stan_data", log_dir)
        group_data = load_obj('group_data', log_dir)

    ############################################################
    ################ Compile Model or Load ##########
    path_to_stan = './codebase/stan_code/'+str(model_path)+'/'

    if stan_model == 0:
        with open(path_to_stan+'saturated_2.stan', 'r') as file:
            model_code = file.read()
        param_names = ['alpha', 'full_sigma', 'sigma', 'Marg_cov',
                    'Marg_cov_cont', 'z', 'yy_latent']
    elif stan_model == 1:  # simplest model - parametrization 2
        with open(path_to_stan+'factor_model_1.stan', 'r') as file:
            model_code = file.read()
        param_names = ['alpha', 'beta', 'theta', 'yy_latent', 'Marg_cov_cont']
    elif stan_model == 2:  # Omega cov for bin only
        with open(path_to_stan+'factor_model_3.stan', 'r') as file:
            model_code = file.read()
        param_names = ['alpha', 'beta',
                    'Omega_cov', 'Phi_cov', 'theta', 'yy_latent',
                    'Marg_cov_cont']
    else:
        print("model option should be 1 or 2")

    if bool(print_model):
        print(model_code)
    file = open(log_dir+"model.txt", "w")
    file.write(model_code)
    file.close()
    if not gen_model:
        with open('log/compiled_models/'+str(model_path)+'/model%s/model.txt' % stan_model, 'r') as file:
            saved_model = file.read()
        if saved_model == model_code:
            sm = load_obj('sm', 'log/compiled_models/'+str(model_path)+'/model%s/' % stan_model)
            if stan_model == 0:
                param_names = ['alpha', 'full_sigma', 'sigma', 'Marg_cov',
                            'Marg_cov_cont', 'z', 'yy_latent']
            elif stan_model == 1:
                param_names = ['alpha', 'beta', 'theta',
                            'yy_latent', 'Marg_cov_cont']
            elif stan_model == 2:
                param_names = ['alpha', 'beta',
                            'Omega_cov', 'Phi_cov', 'theta', 'yy_latent',
                            'Marg_cov_bin', 'Marg_cov_cont']
            else:
                print("model option should be 1 or 2")
    else:
        print("\n\nCompiling model")
        sm = pystan.StanModel(model_code=model_code, verbose=False)
        try:
            print("\n\nSaving compiled model in directory %s" % log_dir)
            save_obj(sm, 'sm', 'log/compiled_models/'+str(model_path)+'/model%s/' % stan_model)
            file = open('log/compiled_models/'+str(model_path)+'/model%s/model.txt' %
                        stan_model, "w")
            file.write(model_code)
            file.close()
        except:
            # Print error message
            print("could not save the stan model")
    ############################################################
    ################ Fit Model ##########
    print("\n\nFitting model.... \n\n")

    if ppp_cv == 'ppp':  # run PPP
        ps_all_groups = dict()
        fit_run = sm.sampling(
            data=stan_data,
            iter=num_samples + num_warmup,
            warmup=num_warmup,
            chains=num_chains,
            n_jobs=4,
            control={'max_treedepth': 15, 'adapt_delta': 0.99})
        # init = 0)
        try:
            print(
                "\n\nSaving fitted model in directory" % log_dir)
            save_obj(fit_run, 'fit', log_dir)
        except:
            # Print error message
            print("could not save the fit object")

            # return a dictionary of arrays
            group_samples = fit_run.extract(
                permuted=False, pars=param_names)

            ps_all_groups = group_samples
        save_obj(ps_all_groups, 'ps_all_groups', log_dir)
        print("\n\nSaving posterior samples in %s " % log_dir)


    elif ppp_cv == 'cv':  # run CV
        print("\n\nKfold Fitting starts.... \n\n")

        ps_all_groups = dict()
        
        fit_runs = dict()
        for fold_index in range(n_splits):
            print("\n\nFitting model.... \n\n")

            fit_runs[fold_index] = sm.sampling(
                data=stan_data[fold_index]['train'],
                iter=num_samples + num_warmup,
                warmup=num_warmup, chains=num_chains,
                n_jobs=num_chains,
                control={'max_treedepth': 15, 'adapt_delta': 0.99}, init=0)
            try:
                print("\n\nSaving fitted model in directory %s" % log_dir)
                save_obj(fit_runs, 'fit',log_dir)
            except:
                # Print error message
                print("could not save the fit object")
        group_samples = dict()
        for fold_index in range(n_splits):
            print("\n\nSaving posterior for fold %s samples in %s" %
                    (fold_index, log_dir))
            # return a dictionary of arrays
            group_samples[fold_index] = fit_runs[fold_index].extract(
                permuted=False, pars=param_names)

            ps_all_groups = group_samples
        save_obj(ps_all_groups, 'ps_all_groups', log_dir)
        print("\n\nSaving posterior samples in %s" % log_dir)

    else:
        print("-cv needs to be 'ppp' or 'cv'")

