import sys, pickle, os
from mpi4py import MPI
from funtofem import *

# import the case utils from the local src/ directory
sys.path.append("src/")
from model_and_solver import make_model_and_solver
from get_argparser import get_argparser

# get argparse
parser = get_argparser()
args = parser.parse_args()

comm = MPI.COMM_WORLD
base_dir = os.path.dirname(os.path.abspath(__file__))

# setup the model and solver
ML_str = "ML" if args.useML else "CF"
f2f_model, solvers, transfer_settings = make_model_and_solver(
    comm, args, model_name=f"{ML_str}_oneway-sizing"
)

# make the trim and sizing driver
tacs_driver = OnewayStructDriver.prime_loads_from_file(
    filename="cfd/loads/uncoupled_turb_loads.txt",
    solvers=solvers,
    model=f2f_model,
    nprocs=args.procs,
    transfer_settings=transfer_settings,
    init_transfer=True,
)

f2f_model.read_design_variables_file(comm, "design/ML-sizing.txt" if args.useML else "design/CF-sizing.txt")

import time
start_time = time.time()
tacs_driver.solve_forward()
fwd_dt = time.time() - start_time
if comm.rank == 0:
    print(f"{fwd_dt=:.4e}")

# print out the function values
if comm.rank == 0:
    for func in f2f_model.get_functions(optim=True):
        print(f"func {func.name} = {func.value.real}")
