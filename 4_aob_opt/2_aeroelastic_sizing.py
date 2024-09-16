"""
Sean P. Engelstad, Georgia Tech 2023

Local machine optimization for the panel thicknesses using nribs-1 OML panels and nribs-1 LE panels
"""

from funtofem import *
import time
from pyoptsparse import SNOPT, Optimization
import numpy as np
import argparse

# import openmdao.api as om
from mpi4py import MPI
from tacs import caps2tacs
import os

parent_parser = argparse.ArgumentParser(add_help=False)
# note there's a weird bug where total # of procs needs to match
# #TACS procs with FUN3D
parent_parser.add_argument("--procs", type=int, default=48)
parent_parser.add_argument("--hotstart", type=bool, default=False)
parent_parser.add_argument("--useML", type=bool, default=False)
parent_parser.add_argument("--deriv", type=bool, default=False)
args = parent_parser.parse_args()

if args.useML:
    from _gp_callback import gp_callback_generator

    model_name = "ML-AE"
else:
    from _closed_form_callback import closed_form_callback as callback

    model_name = "CF-AE"


comm = MPI.COMM_WORLD

base_dir = os.path.dirname(os.path.abspath(__file__))
csm_path = os.path.join(base_dir, "geometry", "gbm-half.csm")

# F2F MODEL and SHAPE MODELS
# ----------------------------------------

f2f_model = FUNtoFEMmodel(model_name)
tacs_model = caps2tacs.TacsModel.build(
    csm_file=csm_path,
    comm=comm,
    problem_name="capsStruct2",
    active_procs=[0],
    verbosity=1,
)
tacs_model.mesh_aim.set_mesh(  # need a refined-enough mesh for the derivative test to pass
    edge_pt_min=2,
    edge_pt_max=50,
    global_mesh_size=0.03,  # 0.3
    max_surf_offset=0.2,
    max_dihedral_angle=15,
).register_to(
    tacs_model
)
f2f_model.structural = tacs_model

tacs_aim = tacs_model.tacs_aim
tacs_aim.set_config_parameter("view:flow", 0)
tacs_aim.set_config_parameter("view:struct", 1)
tacs_aim.set_config_parameter("fullRootRib", 0)

egads_aim = tacs_model.mesh_aim

if comm.rank == 0:
    aim = egads_aim.aim
    aim.input.Mesh_Sizing = {
        "chord": {"numEdgePoints": 20},
        "span": {"numEdgePoints": 10},
        "vert": {"numEdgePoints": 10},
    }


# BODIES AND STRUCT DVs
# -------------------------------------------------

wing = Body.aeroelastic("wing", boundary=2)
# aerothermoelastic

# setup the material and shell properties
nribs = int(tacs_model.get_config_parameter("nribs"))
nOML = nribs - 1
null_material = caps2tacs.Orthotropic.null().register_to(tacs_model)

# create the design variables by components now
# since this mirrors the way TACS creates design variables
component_groups = [f"rib{irib}" for irib in range(1, nribs + 1)]
for prefix in ["spLE", "spTE", "uOML", "lOML"]:
    component_groups += [f"{prefix}{iOML}" for iOML in range(1, nOML + 1)]
component_groups = sorted(component_groups)

# now that you have the list of tacs components, you can build the custom gp callback if using ML case
if args.useML:
    callback = gp_callback_generator(component_groups)

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
    Variable.structural(
        f"{comp}-" + TacsSteadyInterface.LENGTH_VAR, value=panel_length
    ).set_bounds(
        lower=0.0,
        scale=1.0,
        state=True,  # need the length & width to be state variables
    ).register_to(
        wing
    )

    # stiffener pitch variable
    Variable.structural(f"{comp}-spitch", value=0.20).set_bounds(
        lower=0.05, upper=0.5, scale=1.0
    ).register_to(wing)

    # panel thickness variable, shortened DV name for ESP/CAPS, nastran requirement here
    Variable.structural(f"{comp}-T", value=0.02).set_bounds(
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

    Variable.structural(
        f"{comp}-" + TacsSteadyInterface.WIDTH_VAR, value=panel_length
    ).set_bounds(
        lower=0.0,
        scale=1.0,
        state=True,  # need the length & width to be state variables
    ).register_to(
        wing
    )

caps2tacs.PinConstraint("root", dof_constraint=246).register_to(tacs_model)
caps2tacs.PinConstraint("sob", dof_constraint=13).register_to(tacs_model)

# caps2tacs.PinConstraint("root", dof_constraint=123).register_to(tacs_model)

# register the wing body to the model
wing.register_to(f2f_model)

# INITIAL STRUCTURE MESH, SINCE NO STRUCT SHAPE VARS
# --------------------------------------------------

tacs_aim.setup_aim()
tacs_aim.pre_analysis()

# SCENARIOS
# ----------------------------------------------------

T_cruise = 220.550
q_cruise = 10.3166e3

T_sl = 288.15
q_sl = 14.8783e3

# Modified pull-up maneuver conditions
T_mod = 268.338
q_mod = 10.2319e3

# 2 - sea level pull up maneuver
pull_up = Scenario.steady(
    "pull_up",
    steps=400,  # 400
    forward_coupling_frequency=60,  # max 1000 forward steps
    adjoint_steps=35,
    adjoint_coupling_frequency=60,  # max 6000 adjoint steps
    uncoupled_steps=200,  # 300
)
pull_up.set_stop_criterion(
    early_stopping=True,
    min_forward_steps=int(3000 / 60),  # 100 uncoupled + 10 coupled
    min_adjoint_steps=int(1200 / 60),
    post_tight_forward_steps=0,
    post_tight_adjoint_steps=0,
    post_forward_coupling_freq=1,
    post_adjoint_coupling_freq=1,
)

clift = Function.lift().register_to(pull_up)
pull_up_ks = (
    Function.ksfailure(ks_weight=10.0, safety_factor=1.5)
    .optimize(scale=1.0, upper=1.0, objective=False, plot=True, plot_name="ks-cruise")
    .register_to(pull_up)
)
mass_wingbox = (
    Function.mass()
    .optimize(scale=1.0e-3, objective=True, plot=True, plot_name="mass")
    .register_to(pull_up)
)

qfactor = 1.0
aoa_pull_up = pull_up.get_variable("AOA").set_bounds(
    lower=0.0, value=9.0, upper=13, scale=10
)
pull_up.set_temperature(T_ref=T_sl, T_inf=T_sl)
pull_up.set_flow_ref_vals(qinf=qfactor * q_sl)
pull_up.register_to(f2f_model)

# COMPOSITE FUNCTIONS
# -------------------------------------------------------

# pull up load factor constraint
mass_wing = 10.147 * mass_wingbox ** 0.8162  # Elham regression model
mass_payload = 14.5e3  # kg
mass_frame = 25e3  # kg
mass_fuel_res = 2e3  # kg
LGM = mass_payload + mass_frame + mass_fuel_res + 2 * mass_wing
LGW = 9.81 * LGM  # kg => N
pull_up_lift = clift * 2 * q_sl  # already multiplied by area
mod_lift = (
    1.5  # for numerical case, just lower weight of vehicle so AOA settles at 8,9 deg
)
pull_up_LF = mod_lift * pull_up_lift - 2.5 * LGW
pull_up_LF.set_name("pull_up_LF").optimize(
    lower=0.0, upper=0.0, scale=1e-3, objective=False, plot=True
).register_to(f2f_model)

# skin thickness adjacency constraints
variables = f2f_model.get_variables()
adjacency_scale = 10.0
thick_adj = 2.5e-3

comp_groups = ["spLE", "spTE", "uOML", "lOML"]
comp_nums = [nOML for i in range(len(comp_groups))]
adj_types = ["T"]
adj_types += ["sthick", "sheight"]
adj_values = [thick_adj, thick_adj, 10e-3]
for igroup, comp_group in enumerate(comp_groups):
    comp_num = comp_nums[igroup]
    for icomp in range(1, comp_num):
        # no constraints across sob (higher stress there)
        for iadj, adj_type in enumerate(adj_types):
            adj_value = adj_values[iadj]
            name = f"{comp_group}{icomp}-{adj_type}"
            # print(f"name = {name}", flush=True)
            left_var = f2f_model.get_variables(f"{comp_group}{icomp}-{adj_type}")
            right_var = f2f_model.get_variables(f"{comp_group}{icomp+1}-{adj_type}")
            # print(f"left var = {left_var}, right var = {right_var}")
            adj_constr = left_var - right_var
            adj_constr.set_name(f"{comp_group}{icomp}-adj_{adj_type}").optimize(
                lower=-adj_value, upper=adj_value, scale=10.0, objective=False
            ).register_to(f2f_model)

    for icomp in range(1, comp_num + 1):
        skin_var = f2f_model.get_variables(f"{comp_group}{icomp}-T")
        sthick_var = f2f_model.get_variables(f"{comp_group}{icomp}-sthick")
        sheight_var = f2f_model.get_variables(f"{comp_group}{icomp}-sheight")
        spitch_var = f2f_model.get_variables(f"{comp_group}{icomp}-spitch")

        # stiffener - skin thickness adjacency here
        adj_value = thick_adj
        adj_constr = skin_var - sthick_var
        adj_constr.set_name(f"{comp_group}{icomp}-skin_stiff_T").optimize(
            lower=-adj_value, upper=adj_value, scale=10.0, objective=False
        ).register_to(f2f_model)

        # minimum stiffener spacing pitch > 2 * height
        min_spacing_constr = spitch_var - 2 * sheight_var
        min_spacing_constr.set_name(f"{comp_group}{icomp}-sspacing").optimize(
            lower=0.0, scale=1.0, objective=False
        ).register_to(f2f_model)

        # minimum stiffener AR
        min_stiff_AR = sheight_var - 2.0 * sthick_var
        min_stiff_AR.set_name(f"{comp_group}{icomp}-minstiffAR").optimize(
            lower=0.0, scale=1.0, objective=False
        ).register_to(f2f_model)

        # maximum stiffener AR (for regions with tensile strains where crippling constraint won't be active)
        max_stiff_AR = sheight_var - 8.0 * sthick_var
        max_stiff_AR.set_name(f"{comp_group}{icomp}-maxstiffAR").optimize(
            upper=0.0, scale=1.0, objective=False
        ).register_to(f2f_model)

# DISCIPLINE INTERFACES AND DRIVERS
# -----------------------------------------------------
solvers = SolverManager(comm)

solvers.flow = Fun3d14Interface(
    comm,
    f2f_model,
    fun3d_dir="cfd",
    adjoint_options={"getgrad": True, "outer_loop_krylov": True},
    # adjoint_options={"getgrad" : True},
    forward_stop_tolerance=5e-13,
    forward_min_tolerance=1e-10,  # 1e-10
    adjoint_stop_tolerance=5e-12,
    adjoint_min_tolerance=1e-7,
    debug=False,
)

solvers.structural = TacsSteadyInterface.create_from_bdf(
    model=f2f_model,
    comm=comm,
    nprocs=args.procs,
    bdf_file=tacs_aim.root_dat_file,
    # bdf_file="tacs.dat",
    prefix=tacs_aim.root_analysis_dir,
    callback=callback,
    panel_length_dv_index=0,
    panel_width_dv_index=5,
)

# read in aero loads
aero_loads_file = os.path.join(os.getcwd(), "cfd", "loads", "uncoupled_turb_loads.txt")

# transfer_settings = TransferSettings(npts=50, beta=0.1)
transfer_settings = TransferSettings(npts=200)

# build the funtofem nlbgs coupled driver from the file
f2f_driver = FUNtoFEMnlbgs(
    solvers=solvers,
    transfer_settings=transfer_settings,
    model=f2f_model,
    debug=False,
    reload_funtofem_states=False,
)

if args.deriv:  # test using the finite difference test
    # load the previous design
    # design_in_file = os.path.join(base_dir, "design", "sizing-oneway.txt")
    # f2f_model.read_design_variables_file(comm, design_in_file)

    start_time = time.time()

    # run the finite difference test
    max_rel_error = TestResult.derivative_test(
        "gbm-oneway",
        model=f2f_model,
        driver=f2f_driver,
        status_file="2-derivs.txt",
        complex_mode=False,
        epsilon=1e-4,
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
design_in_file = os.path.join(
    base_dir, "design", "ML-sizing.txt" if args.useML else "CF-sizing.txt"
)
design_out_file = os.path.join(
    base_dir, "design", "ML-AE.txt" if args.useML else "CF-AE.txt"
)

design_folder = os.path.join(base_dir, "design")
if not os.path.exists(design_folder) and comm.rank == 0:
    os.mkdir(design_folder)
history_file = os.path.join(design_folder, "ML-AE.hst" if args.useML else "CF-AE.hst")

# reload previous design
f2f_model.read_design_variables_file(comm, design_in_file)

# change dv bounds relative to previous
for var in f2f_model.get_variables():
    var.upper = var.value * 1.3
    var.lower = var.value / 1.3

manager = OptimizationManager(
    f2f_driver,
    design_out_file=design_out_file,
    hot_start=args.hotstart,
    debug=False,
    hot_start_file=history_file,
    sparse=True,
)

# create the pyoptsparse optimization problem
opt_problem = Optimization("gbm-AE", manager.eval_functions)

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
        "Major iterations limit": 4000,
        "Minor iterations limit": 150000000,
        "Iterations limit": 100000000,
        # "Major step limit": 5e-2, # had this off I think (but this maybe could be on)
        "Nonderivative linesearch": True,  # turns off derivative linesearch
        "Linesearch tolerance": 0.9,
        "Difference interval": 1e-6,
        "Function precision": 1e-10,
        "New superbasics limit": 2000,
        "Penalty parameter": 1.0,  # had this off for faster opt in the single panel case
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
    storeHistory=history_file,  # None
    hotStart=history_file if args.hotstart else None,
)

# print final solution
sol_xdict = sol.xStar
if comm.rank == 0:
    print(f"Final solution = {sol_xdict}", flush=True)
