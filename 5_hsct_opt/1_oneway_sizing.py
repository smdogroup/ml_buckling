import sys, pickle, os
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
