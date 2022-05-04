from codebase.file_utils import save_obj
from codebase.classes_data import Data
import argparse
from codebase.file_utils import make_folder, path_backslash
from run_smclvm import run_smclvm
from pdb import set_trace

parser = argparse.ArgumentParser()
parser.add_argument(
    "-th", "--task_handle", help="hande for task", type=str, default="_"
)
parser.add_argument(
    "-xdir",
    "--existing_directory",
    help="refit compiled model in existing directory",
    type=str,
    default=None,
)
parser.add_argument("-gm", "--gen_model", help="generate model", type=bool, default=0)
parser.add_argument("-simcase", "--simcase", help="number of particles", type=int, default=0)
parser.add_argument("-size", "--size", help="number of particles", type=int, default=100)
parser.add_argument("-run_init_mcmc", "--run_init_mcmc", help="run MCMC to initial th particles", type=bool, default=False)
parser.add_argument("-init_t", "--init_t", help="number of initial data points for MCMC", type=int, default=100)

args = parser.parse_args()


if args.existing_directory is None:
    log_dir = make_folder(args.task_handle)
    print("\n\nCreating new directory: %s" % log_dir)

else:
    log_dir = args.existing_directory
    log_dir = path_backslash(log_dir)
    print("\n\nReading from existing directory: %s" % log_dir)

exp_data = Data(name='seq_data')
exp_data.generate(sim_case=args.simcase)
save_obj(exp_data, 'group_data', log_dir)

smclvm = run_smclvm(
    exp_data = exp_data,
    init_t = args.init_t,
    size = args.size,
    gen_model = args.gen_model,
    log_dir = log_dir,
    degeneracy_limit=0.5,
    name="ibis_lvm",
    run_init_mcmc=args.run_init_mcmc
    )