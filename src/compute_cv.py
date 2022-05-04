import numpy as np
import pandas as pd
import pystan
import os
from codebase.file_utils import save_obj, load_obj
from codebase.model_fit_cv_combined import get_combined_logscore
from codebase.post_process import remove_cn_dimension

# from codebase.model_fit_bin import get_lgscr_bin
import argparse
from copy import deepcopy
from pdb import set_trace

parser = argparse.ArgumentParser()
parser.add_argument("logdir", help="path to files", type=str, default=None)
parser.add_argument(
    "stan_model", help="", type=int, default=1)

# Optional arguments
parser.add_argument(
    "-nsim",
    "--nsim_cv",
    help="number of posterior samples to use for CV",
    type=int,
    default=1000,
)

args = parser.parse_args()

log_dir = args.logdir
if log_dir[-1] != "/":
    log_dir = log_dir + "/"

data = load_obj("stan_data", log_dir)
ps = load_obj("ps_all_groups", log_dir)

num_of_folds = len(ps.keys())
for name in ps[0].keys():
    for fold_index in range(num_of_folds):
        ps[fold_index][name] = remove_cn_dimension(ps[fold_index][name])

Ds = dict()
for fold_index in range(num_of_folds):
    Ds[fold_index] = get_combined_logscore(
        ps=ps[fold_index],
        data = data[fold_index]["test"],
        nsim= args.nsim_cv,
        model_num = args.stan_model)
    
a = [Ds[fold] for fold in range(num_of_folds)]
print("\nSum over Folds %.2f" % (np.sum(a)))