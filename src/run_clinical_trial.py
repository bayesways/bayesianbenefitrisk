import numpy as np
import pandas as pd
import datetime
import sys
import os
from codebase.file_utils import save_obj, load_obj
from codebase.real_data_fit import run_fit_for_real_data, compute_results
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "num_warmup", help="number of warm up iterations", type=int, default=1000
)
parser.add_argument(
    "num_samples", help="number of post-warm up iterations", type=int, default=1000
)
parser.add_argument("stan_model", help="", type=int, default=1)
# Optional arguments
parser.add_argument(
    "-cov_num", "--cov_num", help="number of covariance structure", type=int, default=1
)
parser.add_argument("-pool", "--cov_pool", help="pool cov matrix", type=int, default=1)
parser.add_argument(
    "-gd", "--gen_data", help="gen fresh data", type=bool, default=False
)
parser.add_argument("-data_sim", "--data_sim", help="data to use", type=int, default=0)
parser.add_argument(
    "-gm", "--gen_model", help="generate model", type=bool, default=False
)
parser.add_argument(
    "-num_chains", "--num_chains", help="number of MCMC chains", type=int, default=1
)
parser.add_argument(
    "-th", "--task_handle", help="hande for task", type=str, default="_"
)
parser.add_argument(
    "-prm", "--print_model", help="print model on screen", type=int, default=0
)
parser.add_argument(
    "-xdir",
    "--existing_directory",
    help="refit compiled model in existing directory",
    type=str,
    default=None,
)
parser.add_argument("-nfl", "--n_splits", help="number of folds", type=int, default=2)
parser.add_argument("-cv", "--ppp_cv", help="run PPP or CV", type=str, default="ppp")
parser.add_argument(
    "-cvseed", "--cv_random_seed", help="random seed for CV split", type=int, default=27
)

args = parser.parse_args()

############################################################
###### Create Directory or Open existing ##########
if args.existing_directory is None:
    nowstr = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_")  # ISO 8601 format
    log_dir = (
        "./log/"
        + nowstr
        + "%s_%s_%s_%s/"
        % (args.task_handle, args.cov_num, args.ppp_cv, args.stan_model)
    )
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    print("\n\nCreating new directory: %s" % log_dir)

else:
    log_dir = args.existing_directory
    if log_dir[-1] != "/":
        print("\n\nAppending `/`-character at the end of directory")
        log_dir = log_dir + "/"
    print("\n\nReading from existing directory: %s" % log_dir)


run_fit_for_real_data(
    log_dir,
    existing_directory=args.existing_directory,
    stan_model=args.stan_model,
    gen_data=args.gen_data,
    gen_model=args.gen_model,
    data_sim=args.data_sim,
    cov_num=args.cov_num,
    ppp_cv=args.ppp_cv,
    n_splits=args.n_splits,
    cv_random_seed=args.cv_random_seed,
    print_model=args.print_model,
    num_samples=args.num_samples,
    num_warmup=args.num_warmup,
    num_chains=args.num_chains,
)
