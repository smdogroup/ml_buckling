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
    comm, args, model_name=f"{ML_str}_oneway-trim-sizing"
)

# load the initial trim dictionary for lift, AOA values
initial_trim_dict = None
if comm.rank == 0:
    with open(f'cfd/loads/initial_trim_dict.pkl', 'rb') as f:
        initial_trim_dict = pickle.load(f)
initial_trim_dict = comm.bcast(initial_trim_dict, root=0)

# if args.reload_design:
# f2f_model.read_design_variables_file(comm, "design/CF-sizing_design.txt")
f2f_model.read_design_variables_file(comm, "design/ML-sizing_design.txt")

# make the trim and sizing driver
tacs_driver = OnewayStructTrimDriver.prime_loads_from_file(
    filename="cfd/loads/coupled-loads.txt",
    solvers=solvers,
    model=f2f_model,
    initial_trim_dict=initial_trim_dict,
    nprocs=args.procs,
    transfer_settings=transfer_settings,
    init_transfer=True,
)

tacs_driver.solve_forward()

for func in f2f_model.get_functions(optim=True):
    if "ksfailure" in func.name and comm.rank == 0:
        print(f"{func.full_name=} {func.value=}")