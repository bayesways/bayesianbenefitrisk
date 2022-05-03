import numpy as np
import pandas as pd
import pystan
import os
from codebase.file_utils import save_obj, load_obj
from codebase.model_fit_cont import get_PPP as get_PPP_cont
from codebase.post_process import remove_cn_dimension
from codebase.model_fit_bin import get_PPP as get_PPP_bin
import argparse
from copy import deepcopy
from pdb import set_trace

parser = argparse.ArgumentParser()
parser.add_argument("logdir", help="path to files", type=str, default=None)

# Optional arguments
parser.add_argument(
    "-nsim",
    "--nsim_ppp",
    help="number of posterior samples to use for PPP",
    type=int,
    default=500,
)

args = parser.parse_args()

print(args)

log_dir = args.logdir
if log_dir[-1] != "/":
    log_dir = log_dir + "/"


print("\n\nChecking data integrity...\n\n")
data = load_obj("stan_data", log_dir)
ps = load_obj("ps_all_groups", log_dir)

for name in ["alpha_c", "alpha_b", "Marg_cov_cont", "yy"]:
    ps[name] = remove_cn_dimension(ps[name])

#### Continuous part
print("Continuous Data")
ps = deepcopy(ps)

ind_alpha_b = ps["alpha_b"][:, (data["grp"] - 1), :]
ps["alpha_b"] = ind_alpha_b
ind_alpha_c = ps["alpha_c"][:, (data["grp"] - 1), :]
ps["alpha_c"] = ind_alpha_c

ind_Marg_cov = ps["Marg_cov_cont"][:, (data["grp"] - 1), :, :]
ps["Marg_cov_cont"] = ind_Marg_cov

# make continuous data
data_cont = data.copy()
data_cont["y"] = data_cont["yc"]

PPP_vals = get_PPP_cont(data_cont, ps, args.nsim_ppp)

ppp_cn = 100 * np.sum(PPP_vals[:, 0] < PPP_vals[:, 1]) / args.nsim_ppp
print(ppp_cn)

## Binarty part
print("\n\n\nBinary Data")
# make binary data
ps_bin = deepcopy(ps)
data_bin = data.copy() 
data_bin["D"] = data_bin["yb"].astype(int)
if ps_bin["yy"].shape[-1] > data_bin["Kb"]:
    ps_bin["yy"] = ps["yy"][:, :, data_bin["Kc"] :]

PPP_vals = get_PPP_bin(data_bin, ps_bin, args.nsim_ppp)
ppp_cn = 100 * np.sum(PPP_vals[:, 0] < PPP_vals[:, 1]) / args.nsim_ppp

# take the mean sum log-score)
avg_logscore = np.round(ppp_cn)
print(avg_logscore, "\n")
