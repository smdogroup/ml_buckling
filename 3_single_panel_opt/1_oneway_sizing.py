"""
Sean P. Engelstad, Georgia Tech 2023

Local machine optimization for the panel thicknesses using nribs-1 OML panels and nribs-1 LE panels
"""

import time, numpy as np
from pyoptsparse import SNOPT, Optimization
from funtofem import *
from mpi4py import MPI
import os, argparse

parent_parser = argparse.ArgumentParser(add_help=False)
parent_parser.add_argument("--procs", type=int, default=4)
parent_parser.add_argument("--hotstart", type=bool, default=False)
parent_parser.add_argument("--useML", type=bool, default=False)
parent_parser.add_argument("--testDeriv", type=bool, default=False)
args = parent_parser.parse_args()

if args.useML:
    from _gp_callback import gp_callback_generator
    callback = gp_callback_generator(["panel"])
    model_name = "ML-panel"
else:
    from _closed_form_callback import closed_form_callback as callback
    model_name = "CF-panel"

comm = MPI.COMM_WORLD

base_dir = os.path.dirname(os.path.abspath(__file__))

# F2F MODEL and SHAPE MODELS
# ----------------------------------------

f2f_model = FUNtoFEMmodel(model_name)

# BODIES AND STRUCT DVs
# -------------------------------------------------

panel = Body.aeroelastic("panel")
# aerothermoelastic

# create the stiffened panel design variables

# panel length state variable
Variable.structural(TacsSteadyInterface.LENGTH_VAR, value=0.5).set_bounds(
    lower=0.0, scale=1.0, state=True
).register_to(panel)

# stiffener pitch variable
Variable.structural("spitch", value=0.20).set_bounds(
    lower=0.05, upper=0.5, scale=1.0
).register_to(panel)

# panel thickness variable
Variable.structural("pthick", value=0.02).set_bounds(
    lower=0.002, upper=0.1, scale=100.0
).register_to(panel)

# stiffener height
sheight = Variable.structural("sheight", value=0.05).set_bounds(
    lower=0.002, upper=0.1, scale=10.0
).register_to(panel)

# stiffener thickness
sthick = Variable.structural("sthick", value=0.02).set_bounds(
    lower=0.002, upper=0.1, scale=100.0
).register_to(panel)

# panel width variable
Variable.structural(TacsSteadyInterface.WIDTH_VAR, value=0.5).set_bounds(
    lower=0.0, scale=1.0, state=True
).register_to(panel)

# register the wing body to the model
panel.register_to(f2f_model)

# SCENARIOS
# ----------------------------------------------------

# make a funtofem scenario
oneway_struct = Scenario.steady("oneway-struct", steps=2)

# increase ksfailure scale to make it stricter on infeasibility for that.
Function.ksfailure(ks_weight=20.0, safety_factor=1.5).optimize(
    scale=1.0, upper=1.0, objective=False, plot=True
).register_to(oneway_struct)

Function.mass().optimize(
    scale=1.0e-3, objective=True, plot=True
).register_to(oneway_struct)

oneway_struct.register_to(f2f_model)

# COMPOSITE FUNCTIONS
# ----------------------------------------------------
# in one design step tends to drop sheight too low too fast and stays there
# trying out a stiffener AR min constraint, 
#    don't need max one bc of stiffener crippling

stiffARconstr = sheight - 2 * sthick
stiffARconstr.set_name("stiffAR").optimize(
    lower=0.0, scale=1.0, objective=False
).register_to(f2f_model)


# DISCIPLINE INTERFACES AND DRIVERS
# -----------------------------------------------------
solvers = SolverManager(comm)
# solvers.flow = Fun3dInterface(comm, f2f_model, fun3d_dir="meshes")
solvers.structural = TacsSteadyInterface.create_from_bdf(
    model=f2f_model,
    comm=comm,
    nprocs=args.procs,
    bdf_file="_plate.bdf",
    prefix="_struct", # baseline file path is current so empty string
    callback=callback,
    panel_length_dv_index=0,
    panel_width_dv_index=5,
)

# aero loads transfer involves zero aero loads to zero struct loads (so settings don't really matter here)
transfer_settings = TransferSettings(npts=10)

# build the shape driver from the file
# no fixed aero loads here, just fixed struct loads
tacs_driver = OnewayStructDriver(
    solvers=solvers,
    model=f2f_model,
    nprocs=args.procs,
    transfer_settings=transfer_settings,
)


if args.testDeriv:  # test using the finite difference test
    # load the previous design
    # design_in_file = os.path.join(base_dir, "design", "sizing-oneway.txt")
    # f2f_model.read_design_variables_file(comm, design_in_file)

    start_time = time.time()

    # run the finite difference test
    max_rel_error = TestResult.derivative_test(
        "oneway-panel-CF",
        model=f2f_model,
        driver=tacs_driver,
        status_file="1-CF-derivs.txt",
        complex_mode=False,
        epsilon=1e-5,
    )

    end_time = time.time()
    dt = end_time - start_time
    if comm.rank == 0:
        print(f"total time for ssw derivative test is {dt} seconds", flush=True)
        print(f"max rel error = {max_rel_error}", flush=True)

    # exit before optimization
    exit()

# PYOPTSPARSE OPTMIZATION
# -------------------------------------------------------------

# create an OptimizationManager object for the pyoptsparse optimization problem
# design_in_file = os.path.join(base_dir, "design", "sizing.txt")
design_out_file = os.path.join(base_dir, "design", "ML-sizing.txt" if args.useML else "CF-sizing.txt")

design_folder = os.path.join(base_dir, "design")
if not os.path.exists(design_folder) and comm.rank == 0:
    os.mkdir(design_folder)
history_file = os.path.join(design_folder, "sizing.hst")
store_history = True
store_history_file = history_file if store_history else None
hot_start_file = history_file if args.hotstart else None

# reload previous design
# not needed since we are hot starting
# f2f_model.read_design_variables_file(comm, design_out_file)

manager = OptimizationManager(
    tacs_driver,
    design_out_file=design_out_file,
    hot_start=args.hotstart,
    debug=True, # having this flag on means error catching is disabled, so you will see any error messages (although they shouldn't happen anyways)
    hot_start_file=hot_start_file,
    sparse=True,
)

# create the pyoptsparse optimization problem
opt_problem = Optimization("panel-oneway-sizing", manager.eval_functions)

# add funtofem model variables to pyoptsparse
manager.register_to_problem(opt_problem)

# run an SNOPT optimization
snoptimizer = SNOPT(
    options={
        "Print frequency": 1000,
        "Summary frequency": 10000000,
        "Major feasibility tolerance": 1e-6,
        "Major optimality tolerance": 1e-6,
        "Verify level": 0,
        "Major iterations limit": 1000,
        "Minor iterations limit": 150000000,
        "Iterations limit": 100000000,
        # "Major step limit": 5e-2,
        "Nonderivative linesearch": True, # turns off nonderivative linesearches
        "Linesearch tolerance": 0.9,
        "Difference interval": 1e-6,
        "Function precision": 1e-10,
        "New superbasics limit": 2000,
        # "Penalty parameter": 1,
        "Scale option": 1,
        "Hessian updates": 40,
        "Print file": os.path.join("SNOPT_print.out"),
        "Summary file": os.path.join("SNOPT_summary.out"),
    }
)

sol = snoptimizer(
    opt_problem,
    sens=manager.eval_gradients,
    storeHistory=store_history_file,
    hotStart=hot_start_file,
)

# print final solution
sol_xdict = sol.xStar
if comm.rank == 0:
    print(f"Final solution = {sol_xdict}", flush=True)

# print the sheight, sthick and stiffAR constraints and vars
if comm.rank == 0:
   print(f"\n\nsheight = {sheight.value}", flush=True)
   print(f"sthick = {sthick.value}", flush=True)
   print(f"stiffARconstr = {stiffARconstr.value}", flush=True)
