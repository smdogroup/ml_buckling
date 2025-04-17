import sys, pickle, os
import numpy as np
from mpi4py import MPI
from funtofem import *

# import the case utils from the local src/ directory
sys.path.append("src/")
from model_and_solver import make_model_and_solver
from optimize_model import optimize_model
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

# adjusting from AOA = [5.0, -3.0] for pullup, pushdown to:
# AOA = [16.0159633 , -7.64973179])
trim_load_factors = np.array([2.69476358, 4.09982119])
# made mistake using wrong qinf =    instead of 1.0736e4, so nearly half loads for that
# trim load factor still the same but the loads out of FUN3D were about twice as high due to wrong qinf
trim_load_factors *= 1.0736e4 / 2.2657e4
# scale up loads by this

body = f2f_model.bodies[0]
for i, scenario in enumerate(f2f_model.scenarios):
    fa = body.get_aero_loads(scenario)
    fa *= trim_load_factors[i]
tacs_driver._transfer_fixed_aero_loads()

# if args.test_derivs:
#     test_derivs(
#         comm, 
#         f2f_model, 
#         tacs_driver,
#         args=args,
#         base_dir=base_dir,
#         design_in_filename="ML-benchmark-sizing.txt" if args.useML else "CF-benchmark-sizing.txt",
#     )

# run pyoptsparse optimization
sol = optimize_model(
    comm,
    tacs_driver,
    args,
    file_prefix="ML-sizing" if args.useML else "CF-sizing",
    verify_level=0,
    debug=False
)
sol_xdict = sol.xStar
if comm.rank == 0:
    print(f"Final solution = {sol_xdict}", flush=True)
