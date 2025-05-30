import sys, pickle, os
import numpy as np
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
    init_transfer=False,
)

# adjusting from AOA = [5.0, -3.0] for pullup, pushdown to:
# AOA = [16.0159633 , -7.64973179])
trim_load_factors = np.array([2.69476358, 4.09982119])
# made mistake using wrong qinf =    instead of 1.0736e4, so nearly half loads for that
trim_load_factors *= 1.0736e4 / 2.2657e4
# scale up loads by this

body = f2f_model.bodies[0]
for i, scenario in enumerate(f2f_model.scenarios):
    fa = body.get_aero_loads(scenario)
    fa *= trim_load_factors[i]
tacs_driver._transfer_fixed_aero_loads()

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
