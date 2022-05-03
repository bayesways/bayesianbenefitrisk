import os
import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal, norm, bernoulli
from scipy.special import expit, logit
from codebase.tlk import check_posdef


def interweave_df(df):
    indices=dict()
    for name in ['MET', 'AVM', 'RSG']:
        indices[name] = df[df['atrtgrp']==name].index
    min_length = 146

    indices1 = dict()
    for name in ['MET', 'AVM', 'RSG']:
        indices1[name] = indices[name][:146]

    indices2 = dict()
    for name in ['AVM', 'RSG']:
        indices2[name] = indices[name][146:150]


    lists = [indices1['MET'], indices1["AVM"], indices1['RSG']]
    interweave_df_index1 =  [val for tup in zip(*lists) for val in tup]

    interweave_df_index2 = [val for pair in zip(indices2['AVM'], indices2["RSG"]) for val in pair]

    interweave_df_index = interweave_df_index1+interweave_df_index2+ list(indices['RSG'][150:])
    return df.loc[interweave_df_index]


def gen_simulation_data_no_corr(
    nsim_data,
    grpnum,
    J=6,
    Kc = 2,
    random_seed=None,
    diff_grps = 0
    ):
    if random_seed is not None:
        np.random.seed(random_seed)

    if diff_grps == 0:
           alpha = np.array([-1.83, -2.95,  -1.3,  -2.5, -2, -3])

    elif diff_grps == 1: # Grp2 better efficacy and same Adverse Events
        if grpnum == 1:
            alpha = np.array([-1.83, -2.95,  -1.3,  -2.5, -2, -3])
        elif grpnum == 2:
            alpha = np.array([-3.5, -4.5,  -1.3,  -2.5, -2, -3])
        else:
            print('grpnum has to be 1 or 2')
    elif diff_grps == 2: # Grp2 better efficacy and less Adverse Events
        if grpnum == 1:
            alpha = np.array([-1.83, -2.95,  -1.3,  -2.5, -2, -3])
        elif grpnum == 2:
            alpha = np.array([-3.5, -4.5, -1.8,  -3, -2.5, -3.5])
        else:
            print('grpnum has to be 1 or 2')
    elif diff_grps == 3: # Grp2 better efficacy more Adverse Events
        if grpnum == 1:
            alpha = np.array([-1.83, -2.95,  -1.3,  -2.5, -2, -3])
        elif grpnum == 2:
            alpha = np.array([-3.5, -4.5,  -.8,  -2, -1.5, -2.5])
        else:
            print('grpnum has to be 1 or 2')
    else:
        print('diff_grps has to be in [0,3]')

    Phi_cov = np.eye(J)
    theta = np.array([1.2, 2.99])
    Phi_cov[:Kc, :Kc] = np.diag(theta)
    zz = multivariate_normal.rvs(
        mean=alpha,
        cov=Phi_cov,
        size=nsim_data
    )
    
    prob_adverse = expit(zz[:,Kc:])
    yb = bernoulli.rvs(p=prob_adverse)  
    yc = zz[:,:Kc]
    
    data = dict()
    data['random_seed'] = random_seed
    data['N'] = nsim_data
    data['J'] = J
    data['Kc'] = Kc
    data['alpha'] = alpha
    data['Phi_cov'] = Phi_cov
    data['z'] = zz
    data['yb'] = yb
    data['yc'] = yc
    data['grpnum'] = (np.ones(nsim_data)*grpnum).astype(int)
    data['y'] = np.concatenate([yc, yb], axis=1)
    data['p_adv'] = prob_adverse
    return data


def gen_simulation_data_corr(
    nsim_data,
    grpnum,
    J=6,
    Kc = 2,
    rho = 0.,
    random_seed=None,
    diff_grps = 0
    ):
    if random_seed is not None:
        np.random.seed(random_seed)

    if diff_grps == 0:
           alpha = np.array([-1.83, -2.95,  -1.3,  -2.5, -2, -3])

    elif diff_grps == 1: # Grp2 better efficacy and same Adverse Events
        if grpnum == 1:
            alpha = np.array([-1.83, -2.95,  -1.3,  -2.5, -2, -3])
        elif grpnum == 2:
            alpha = np.array([-3.5, -4.5,  -1.3,  -2.5, -2, -3])
        else:
            print('grpnum has to be 1 or 2')
    elif diff_grps == 2: # Grp2 better efficacy and less Adverse Events
        if grpnum == 1:
            alpha = np.array([-1.83, -2.95,  -1.3,  -2.5, -2, -3])
        elif grpnum == 2:
            alpha = np.array([-3.5, -4.5,  -1.8,  -3, -2.5, -3.5])
        else:
            print('grpnum has to be 1 or 2')
    elif diff_grps == 3: # Grp2 better efficacy more Adverse Events
        if grpnum == 1:
            alpha = np.array([-1.83, -2.95,  -1.3,  -2.5, -2, -3])
        elif grpnum == 2:
            alpha = np.array([-3.5, -4.5,  -.8,  -2, -1.5, -2.5])
        else:
            print('grpnum has to be 1 or 2')
    else:
        print('diff_grps has to be in [0,3]')

    Phi_corr = np.eye(J)
    Phi_corr[0, 1]=0.7
    Phi_corr[1, 0]=0.7
    for i in [0,1]:
        for j in [2,3,4,5]:
            if j!=i:
                Phi_corr[i,j]=rho
                Phi_corr[j,i]=Phi_corr[i,j]
    for i in [2,3,4,5]:
        for j in [2,3,4,5]:
            if j!=i:
                Phi_corr[i,j]=0.9
                Phi_corr[j,i]=Phi_corr[i,j]
    theta = np.array([1.2**0.5, 2.99**0.5, 1., 1., 1., 1.])
    Phi_cov = np.diag(theta) @ Phi_corr @  np.diag(theta)
    assert check_posdef(Phi_cov) == 0

    zz = multivariate_normal.rvs(
        mean=alpha,
        cov=Phi_cov,
        size=nsim_data
    )

    prob_adverse = expit(zz[:,Kc:])
    yb = bernoulli.rvs(p=prob_adverse)  
    yc = zz[:,:Kc]
    
    data = dict()
    data['random_seed'] = random_seed
    data['N'] = nsim_data
    data['J'] = J
    data['Kc'] = Kc
    data['alpha'] = alpha
    data['Phi_cov'] = Phi_cov
    data['z'] = zz
    data['yb'] = yb
    data['yc'] = yc
    data['grpnum'] = (np.ones(nsim_data)*grpnum).astype(int)
    data['y'] = np.concatenate([yc, yb], axis=1)
    data['p_adv'] = prob_adverse

    return data


def gen_simulation_data_factor_model(
    nsim_data,
    grpnum,
    J=6,
    K=2,
    rho=0.,
    b1=0.8,
    b2=0.5,
    random_seed=None,
    diff_grps = 0
):
    if random_seed is not None:
        np.random.seed(random_seed)

    beta = np.array([[1, 0],
                     [b1, 0],
                     [0, 1.],
                     [0,  b2],
                     [0,  b2],
                     [0,  b2]], dtype=float)

    if diff_grps == 0:
           alpha = np.array([-1.83, -2.95,  -1.3,  -2.5, -2, -3])

    elif diff_grps == 1: # Grp2 better efficacy and same Adverse Events
        if grpnum == 1:
            alpha = np.array([-1.83, -2.95,  -1.3,  -2.5, -2, -3])
        elif grpnum == 2:
            alpha = np.array([-3.5, -4.5,  -1.3,  -2.5, -2, -3])
        else:
            print('grpnum has to be 1 or 2')
    elif diff_grps == 2: # Grp2 better efficacy and less Adverse Events
        if grpnum == 1:
            alpha = np.array([-1.83, -2.95,  -1.3,  -2.5, -2, -3])
        elif grpnum == 2:
            alpha = np.array([-3.5, -4.5,  -1.8,  -3, -2.5, -3.5])
        else:
            print('grpnum has to be 1 or 2')
    elif diff_grps == 3: # Grp2 better efficacy more Adverse Events
        if grpnum == 1:
            alpha = np.array([-1.83, -2.95,  -1.3,  -2.5, -2, -3])
        elif grpnum == 2:
            alpha = np.array([-3.5, -4.5,  -.8,  -2, -1.5, -2.5])
        else:
            print('grpnum has to be 1 or 2')
    else:
        print('diff_grps has to be in [0,3]')

    Phi_corr = np.eye(K)
    Phi_corr[0, 1] = rho
    Phi_corr[1, 0] = rho
    # Phi_cov = np.diag(sigma_z) @ Phi_corr @  np.diag(sigma_z)
    Phi_cov = Phi_corr

    assert check_posdef(Phi_cov) == 0
    zz = multivariate_normal.rvs(mean=np.zeros(K), cov=Phi_cov,
                                 size=nsim_data)
    
    yy_latent = alpha + zz @ beta.T


    prob_adverse = expit(yy_latent[:,2:])
    yb = bernoulli.rvs(p=prob_adverse)  
        
    theta = np.array([1.2, 2.99])
    
    yc = yy_latent[:,:2] + multivariate_normal.rvs(
        mean=np.zeros(2),
        cov=np.diag(theta**2),
        size=nsim_data
    )
    
        
    data = dict()
    data['random_seed'] = random_seed
    data['N'] = nsim_data
    data['K'] = K
    data['J'] = J
    data['alpha'] = alpha
    data['beta'] = beta
    data['Phi_corr'] = Phi_corr
    data['Phi_cov'] = Phi_cov
    data['z'] = zz
    data['yb'] = yb
    data['yc'] = yc
    data['grpnum'] = (np.ones(nsim_data)*grpnum).astype(int)
    data['y'] = np.concatenate([yc, yb], axis=1)
    data['p_adv'] = prob_adverse
    return data


def create_group_data(nsim, corr, rho=None, diff_grps = 0, seed1=0, seed2=2):
    seeds = [seed1, seed2]
    # seeds = [0,2]
    # seeds = [0,11]

    if corr==0:
        d1 = gen_simulation_data_no_corr(
            nsim,
            1,
            J=6,
            random_seed=seeds[0],
            diff_grps = diff_grps
            )

        d2 = gen_simulation_data_no_corr(
            nsim,
            2,
            J=6,
            random_seed=seeds[1],
            diff_grps = diff_grps
            )
        
        data = dict()
        data['yc'] = np.concatenate([d1['yc'], d2['yc']])
        data['yb'] = np.concatenate([d1['yb'], d2['yb']])
        data['y'] = np.concatenate([d1['y'], d2['y']])
        data['grp'] = np.concatenate([d1['grpnum'], d2['grpnum']])
        data['N'] = d1['N'] + d2['N']
        data['Kc']=d1['Kc']
        data['Kb']=d1['J']-d1['Kc']
        data['K']=d1['J']
        data['rho']=rho
        data['seeds'] = seeds
        data['diff_grps'] = diff_grps
        data['p_adv'] = np.concatenate([d1['p_adv'], d2['p_adv']])
        data['alpha'] = np.vstack([d1['alpha'], d2['alpha']])

    elif corr==1:
        d1 = gen_simulation_data_factor_model(
            nsim,
            1,
            J=6,
            K=2,
            rho=rho,
            b1=0.5,
            b2=0.5,
            random_seed=seeds[0],
            diff_grps = diff_grps
        )

        d2 = gen_simulation_data_factor_model(
            nsim,
            2,
            J=6,
            K=2,
            rho=rho,
            b1=0.5,
            b2=0.5,
            random_seed=seeds[1],
            diff_grps = diff_grps
        )
        
        data = dict()
        data['yc'] = np.concatenate([d1['yc'], d2['yc']])
        data['yb'] = np.concatenate([d1['yb'], d2['yb']])
        data['y'] = np.concatenate([d1['y'], d2['y']])
        data['grp'] = np.concatenate([d1['grpnum'], d2['grpnum']])
        data['N'] = d1['N'] + d2['N']
        data['Kc']=d1['K']
        data['Kb']=d1['J']-d1['K']
        data['K']=d1['J']
        data['rho']=rho
        data['Phi_cov'] = d1['Phi_cov']
        data['seeds'] = seeds
        data['diff_grps'] = diff_grps
        data['p_adv'] = np.concatenate([d1['p_adv'], d2['p_adv']])
        data['alpha'] = np.vstack([d1['alpha'], d2['alpha']])
        data['beta'] = d1['beta']


    elif corr==2:
        d1 = gen_simulation_data_corr(
            nsim,
            1,
            J=6,
            rho = rho,
            random_seed=seeds[0],
            diff_grps = diff_grps
            )

        d2 = gen_simulation_data_corr(
            nsim,
            2,
            J=6,
            rho = rho,
            random_seed=seeds[1],
            diff_grps = diff_grps
            )
        
        data = dict()
        data['yc'] = np.concatenate([d1['yc'], d2['yc']])
        data['yb'] = np.concatenate([d1['yb'], d2['yb']])
        data['y'] = np.concatenate([d1['y'], d2['y']])
        data['grp'] = np.concatenate([d1['grpnum'], d2['grpnum']])
        data['p_adv'] = np.concatenate([d1['p_adv'], d2['p_adv']])
        data['N'] = d1['N'] + d2['N']
        data['Kc']=d1['Kc']
        data['Kb']=d1['J']-d1['Kc']
        data['K']=d1['J']
        data['rho']=rho
        data['Phi_cov'] = d1['Phi_cov']
        data['seeds'] = seeds
        data['diff_grps'] = diff_grps
    else:
        print('corr should be 0 or 1')
        return 0
    return data