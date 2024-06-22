"""
Sean P. Engelstad, Georgia Tech 2023

Local machine optimization for the panel thicknesses using all OML and LE panels (more design variables) but no shape variables
"""

from pyoptsparse import SLSQP, Optimization

# script inputs
hot_start = False
store_history = True

# import openmdao.api as om
from funtofem import *
from mpi4py import MPI
from tacs import caps2tacs
import os
import argparse

parent_parser = argparse.ArgumentParser(add_help=False)
parent_parser.add_argument("--procs", type=int, default=6)
parent_parser.add_argument("--hotstart", type=bool, default=False)
parent_parser.add_argument("--useML", type=bool, default=False)
args = parent_parser.parse_args()

comm = MPI.COMM_WORLD

if args.useML:
    from _gp_callback import gp_callback_generator
    model_name = "ML-oneway"
else:
    from _closed_form_callback import closed_form_callback as callback
    model_name = "CF-oneway"

base_dir = os.path.dirname(os.path.abspath(__file__))
csm_path = os.path.join(base_dir, "geometry", "hsct.csm")

# F2F MODEL and SHAPE MODELS
# ----------------------------------------

f2f_model = FUNtoFEMmodel("hsct-sizing1")
tacs_model = caps2tacs.TacsModel.build(
    csm_file=csm_path,
    comm=comm,
    problem_name="capsStruct1",
    active_procs=[0],
    verbosity=1,
)
tacs_model.mesh_aim.set_mesh(  # need a refined-enough mesh for the derivative test to pass
    edge_pt_min=2,
    edge_pt_max=20,
    global_mesh_size=0.3,
    max_surf_offset=0.2,
    max_dihedral_angle=15,
).register_to(
    tacs_model
)
f2f_model.structural = tacs_model

tacs_aim = tacs_model.tacs_aim
tacs_aim.set_config_parameter("mode:flow", 0)
tacs_aim.set_config_parameter("mode:struct", 1)
tacs_aim.set_config_parameter("wing:allOMLgroups", 1)
tacs_aim.set_config_parameter("wing:includeLE", 0)
tacs_aim.set_config_parameter("wing:includeTE", 0)

for proc in tacs_aim.active_procs:
    if comm.rank == proc:
        aim = tacs_model.mesh_aim.aim
        aim.input.Mesh_Sizing = {
            "chord": {"numEdgePoints": 20},
            "span": {"numEdgePoints": 8},
            "vert": {"numEdgePoints": 4},
        }
        # "LEribFace": {"tessParams": [0.03, 0.1, 3]},
        # "LEribEdge": {"numEdgePoints": 20},

# add tacs constraints in
caps2tacs.PinConstraint("root").register_to(tacs_model)
# caps2tacs.PinConstraint("station2").register_to(tacs_model)
# caps2tacs.TemperatureConstraint("midplane", temperature=0).register_to(tacs_model)

# BODIES AND STRUCT DVs
# -------------------------------------------------

wing = Body.aeroelastic("wing", boundary=5)
# aerothermoelastic

# setup the material and shell properties
null_material = caps2tacs.Orthotropic.null().register_to(tacs_model)

nribs = int(tacs_model.get_config_parameter("wing:nribs"))
nspars = int(tacs_model.get_config_parameter("wing:nspars"))
nOML = nribs - 1

rib_npanels_dict = {}
spar_npanels_dict = {
    1: 2, 2: 4, 3: 5, 4: 7, 5: 9,
    6: 12, 7: 17, 8: 19, 
}
OML_npanels_dict = {
    0: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 4,
    7: 4, 8: 5, 9: 5, 10: 6, 11: 6,  12: 6, 13: 7,
    14: 7, 15: 7, 16: 7, 17: 7, 18: 8, 19: 8, 20:8
}

component_groups = []

for irib in range(1, nribs+1):
    istart = OML_npanels_dict[irib]
    for ichord in range(istart, 9+1):
        component_groups += [f"rib{irib}-{ichord}"]

for iOML in range(1,nOML+1):
    istart = OML_npanels_dict[iOML]
    for ichord in range(istart,9+1):
        component_groups += [f"OMLtop{iOML}-{ichord}", f"OMLbot{iOML}-{ichord}"]

for ispar in range(1, nspars+1):
    ifinal = spar_npanels_dict[ispar]
    for ispan in range(1, ifinal+1):
        component_groups += [f"spar{ispar}-{ispan}"]

for prefix in ["LEspar", "TEspar"]:
    for ispan in range(1, 19+1):
        component_groups += [f"{prefix}-{ispan}"]

# print(f"component groups = {component_groups}")
# exit()

for icomp, comp in enumerate(component_groups):
    caps2tacs.CompositeProperty.null(comp, null_material).register_to(tacs_model)

    # NOTE : need to make the struct DVs in TACS in the same order as the blade callback
    # which is done by components and then a local order

    # panel length variable
    if "rib" in comp:
        panel_length = 0.38
    elif "sp" in comp:
        panel_length = 0.36
    elif "OML" in comp:
        panel_length = 0.65
    Variable.structural(f"{comp}-"+TacsSteadyInterface.LENGTH_VAR, value=panel_length).set_bounds(
        lower=0.0, scale=1.0, state=True, # need the length & width to be state variables
    ).register_to(wing)

    # stiffener pitch variable
    Variable.structural(f"{comp}-spitch", value=0.20).set_bounds(
        lower=0.05, upper=0.5, scale=1.0
    ).register_to(wing)

    # panel thickness variable, shortened DV name for ESP/CAPS, nastran requirement here
    if "rib" in comp:
        panelThick = 0.2
    elif "spar" in comp:
        panelThick = 0.4
    elif "OML" in comp:
        panelThick = 0.1
    Variable.structural(f"{comp}-T", value=panelThick).set_bounds(
        lower=0.002, upper=0.1, scale=100.0
    ).register_to(wing)

    # stiffener height
    Variable.structural(f"{comp}-sheight", value=0.05).set_bounds(
        lower=0.002, upper=0.1, scale=10.0
    ).register_to(wing)

    # stiffener thickness
    Variable.structural(f"{comp}-sthick", value=0.02).set_bounds(
        lower=0.002, upper=0.1, scale=100.0
    ).register_to(wing)

    Variable.structural(f"{comp}-"+TacsSteadyInterface.WIDTH_VAR, value=panel_length).set_bounds(
        lower=0.0, scale=1.0,  state=True, # need the length & width to be state variables
    ).register_to(wing)


# register the wing body to the model
wing.register_to(f2f_model)

# INITIAL STRUCTURE MESH, SINCE NO STRUCT SHAPE VARS
# --------------------------------------------------

tacs_aim.setup_aim()
tacs_aim.pre_analysis()

# SCENARIOS
# ----------------------------------------------------

# make a funtofem scenario
climb = Scenario.steady("climb_turb", steps=350, uncoupled_steps=200)  # 2000
Function.ksfailure(ks_weight=20.0, safety_factor=1.5).optimize(
    scale=1.0, upper=1.0, objective=False, plot=True, plot_name="ks-climb"
).register_to(climb)
Function.mass().optimize(
    scale=1.0e-4, objective=True, plot=True, plot_name="mass"
).register_to(climb)
climb.set_temperature(T_ref=216, T_inf=216)
climb.set_flow_ref_vals(qinf=3.16e4)
climb.register_to(f2f_model)

# COMPOSITE FUNCTIONS
# -------------------------------------------------------
# TBD, this will be a bit tricky here, prob just need some None checks

# DISCIPLINE INTERFACES AND DRIVERS
# -----------------------------------------------------

solvers = SolverManager(comm)
# solvers.flow = Fun3dInterface(comm, f2f_model, fun3d_dir="meshes")
solvers.structural = TacsSteadyInterface.create_from_bdf(
    model=f2f_model,
    comm=comm,
    nprocs=8,
    bdf_file=tacs_aim.root_dat_file,
    prefix=tacs_aim.root_analysis_dir,
)

# read in aero loads
aero_loads_file = os.path.join(os.getcwd(), "cfd", "loads", "uncoupled_turb_loads.txt")

transfer_settings = TransferSettings(npts=200)

# build the shape driver from the file
tacs_driver = OnewayStructDriver.prime_loads_from_file(
    filename=aero_loads_file,
    solvers=solvers,
    model=f2f_model,
    callback=callback,
    nprocs=8,
    transfer_settings=transfer_settings,
)

# PYOPTSPARSE OPTMIZATION
# -------------------------------------------------------------

# create an OptimizationManager object for the pyoptsparse optimization problem
design_out_file = os.path.join(base_dir, "design", "ML-sizing.txt" if args.useML else "CF-sizing.txt")

manager = OptimizationManager(
    tacs_driver, design_out_file=design_out_file, hot_start=hot_start, debug=True
)

# create the pyoptsparse optimization problem
opt_problem = Optimization("hsctOpt", manager.eval_functions)

# add funtofem model variables to pyoptsparse
manager.register_to_problem(opt_problem)

# run an SNOPT optimization

design_folder = os.path.join(base_dir, "design")
if not os.path.exists(design_folder):
    os.mkdir(design_folder)
history_file = os.path.join(design_folder, "ML-sizing.hst" if args.useML else "CF-sizing.hst")
store_history_file = history_file if store_history else None
hot_start_file = history_file if hot_start else None

snoptimizer = SNOPT(
    options={
        "Print frequency": 1000,
        "Summary frequency": 10000000,
        "Major feasibility tolerance": 1e-6,
        "Major optimality tolerance": 1e-6,
        "Verify level": 0,
        "Major iterations limit": 4000,
        "Minor iterations limit": 150000000,
        "Iterations limit": 100000000,
        # "Major step limit": 5e-2, # had this off I think (but this maybe could be on)
        "Nonderivative linesearch": True, # turns off derivative linesearch
        "Linesearch tolerance": 0.9,
        "Difference interval": 1e-6,
        "Function precision": 1e-10,
        "New superbasics limit": 2000,
        "Penalty parameter": 1.0, # had this off for faster opt in the single panel case
        # however ksfailure becomes too large with this off. W/ on merit function goes down too slowly though
        # try intermediate value btw 0 and 1 (smaller penalty)
        # this may be the most important switch to change for opt performance w/ ksfailure in the opt
        # TODO : could try higher penalty parameter like 50 or higher and see if that helps reduce iteration count..
        #   because it often increases the penalty parameter a lot near the optimal solution anyways
        "Scale option": 1,
        "Hessian updates": 40,
        "Print file": os.path.join("SNOPT_print.out"),
        "Summary file": os.path.join("SNOPT_summary.out"),
    }
)

sol = snoptimizer(
    opt_problem,
    sens=manager.eval_gradients,
    storeHistory=history_file, #None
    hotStart=history_file if args.hotstart else None,
)

# print final solution
sol_xdict = sol.xStar
if comm.rank == 0:
    print(f"Final solution = {sol_xdict}", flush=True)
