"""
Sean P. Engelstad, Georgia Tech 2023

Local machine optimization for the panel thicknesses using all OML and LE panels (more design variables) but no shape variables
"""

from funtofem import *
from pyoptsparse import SNOPT, Optimization
import gc  # garbage collection
import psutil, os
import numpy as np

# script inputs
# hot_start = False
store_history = True


def process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1e9  # random shared memory
    # return mem_info.vms # virtual memory storage


mem1 = process_memory()
print(f"initial memory = {mem1:.8e} GB")

# import openmdao.api as om
from mpi4py import MPI
from tacs import caps2tacs
import os, sys
import argparse

parent_parser = argparse.ArgumentParser(add_help=False)
parent_parser.add_argument("--procs", type=int, default=4)  # 128
parent_parser.add_argument('--hotstart', default=False, action=argparse.BooleanOptionalAction)
parent_parser.add_argument('--useML', default=False, action=argparse.BooleanOptionalAction)
parent_parser.add_argument('--newMesh', default=False, action=argparse.BooleanOptionalAction)
args = parent_parser.parse_args()

hot_start = args.hotstart

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

f2f_model = FUNtoFEMmodel(model_name)
# tacs_model = caps2tacs.TacsModel.build(
#    csm_file=csm_path,
#    comm=comm,
#    problem_name="capsStruct1",
#    active_procs=[0],
#    verbosity=1,
# )
# tacs_model.mesh_aim.set_mesh(  # need a refined-enough mesh for the derivative test to pass
#    edge_pt_min=2,
#    edge_pt_max=20,
#    mesh_elements="Mixed",
#    global_mesh_size=0.05,
#    max_surf_offset=0.2,
#    max_dihedral_angle=15,
# ).register_to(
#    tacs_model
# )
# f2f_model.structural = tacs_model

# tacs_aim = tacs_model.tacs_aim
# tacs_aim.set_config_parameter("mode:flow", 0)
# tacs_aim.set_config_parameter("mode:struct", 1)
# tacs_aim.set_config_parameter("wing:allOMLgroups", 1)
# tacs_aim.set_config_parameter("wing:includeLE", 0)
# tacs_aim.set_config_parameter("wing:includeTE", 0)
# tacs_aim.set_config_parameter("wing:nspars", 40) # same # as concorde

# for proc in tacs_aim.active_procs:
#    if comm.rank == proc:
#        aim = tacs_model.mesh_aim.aim
#        aim.input.Mesh_Sizing = {
#            "chord": {"numEdgePoints": 20},
#            "span": {"numEdgePoints": 8},
#            "vert": {"numEdgePoints": 4},
#        }
# "LEribFace": {"tessParams": [0.03, 0.1, 3]},
# "LEribEdge": {"numEdgePoints": 20},

# add tacs constraints in
# caps2tacs.PinConstraint("root").register_to(tacs_model)
# caps2tacs.PinConstraint("station2").register_to(tacs_model)
# caps2tacs.TemperatureConstraint("midplane", temperature=0).register_to(tacs_model)

# BODIES AND STRUCT DVs
# -------------------------------------------------

wing = Body.aeroelastic("wing", boundary=5)
# aerothermoelastic

# setup the material and shell properties
# null_material = caps2tacs.Orthotropic.null().register_to(tacs_model)

# nribs = int(tacs_model.get_config_parameter("wing:nribs"))
# nspars = int(tacs_model.get_config_parameter("wing:nspars"))
# nOML = nribs - 1

# NOTE : the feature in ESP/CAPS that gives you the names of all capsGroups
# failed to list all of them since it hit the character limit in the textbox
# so I basically had to count the ranges of all capsGroups myself..
component_groups = []
for prefix in ["OMLtop", "OMLbot"]:
    component_groups += [f"{prefix}1-{ispar}" for ispar in range(1, 41 + 1)]
    component_groups += [f"{prefix}2-{ispar}" for ispar in range(3, 41 + 1)]
    component_groups += [f"{prefix}3-{ispar}" for ispar in range(6, 41 + 1)]
    component_groups += [f"{prefix}4-{ispar}" for ispar in range(9, 41 + 1)]
    component_groups += [f"{prefix}5-{ispar}" for ispar in range(12, 41 + 1)]
    component_groups += [f"{prefix}6-{ispar}" for ispar in range(15, 41 + 1)]
    component_groups += [f"{prefix}7-{ispar}" for ispar in range(18, 41 + 1)]
    component_groups += [f"{prefix}8-{ispar}" for ispar in range(20, 41 + 1)]
    component_groups += [f"{prefix}9-{ispar}" for ispar in range(23, 41 + 1)]
    component_groups += [f"{prefix}10-{ispar}" for ispar in range(24, 41 + 1)]
    component_groups += [f"{prefix}11-{ispar}" for ispar in range(26, 41 + 1)]
    component_groups += [f"{prefix}12-{ispar}" for ispar in range(27, 41 + 1)]
    component_groups += [f"{prefix}13-{ispar}" for ispar in range(28, 41 + 1)]
    component_groups += [f"{prefix}14-{ispar}" for ispar in range(29, 41 + 1)]
    component_groups += [f"{prefix}15-{ispar}" for ispar in range(30, 41 + 1)]
    component_groups += [f"{prefix}16-{ispar}" for ispar in range(31, 41 + 1)]
    component_groups += [f"{prefix}17-{ispar}" for ispar in range(32, 41 + 1)]
    component_groups += [f"{prefix}18-{ispar}" for ispar in range(33, 41 + 1)]
    component_groups += [f"{prefix}19-{ispar}" for ispar in range(34, 41 + 1)]

# add rib component groups based on OML onesg
rib_groups = ["rib" + comp[6:] for comp in component_groups if "OMLtop" in comp]
rib_groups += [f"rib20-{ispar}" for ispar in range(35, 41 + 1)]
component_groups += rib_groups

# add spar component groups
component_groups += [f"LEspar-{iOML}" for iOML in range(1, 19 + 1)]
component_groups += [f"TEspar-{iOML}" for iOML in range(1, 19 + 1)]
component_groups += [f"spar1-{iOML}" for iOML in range(1, 1 + 1)]
component_groups += [f"spar2-{iOML}" for iOML in range(1, 1 + 1)]
component_groups += [f"spar3-{iOML}" for iOML in range(1, 2 + 1)]
component_groups += [f"spar4-{iOML}" for iOML in range(1, 2 + 1)]
component_groups += [f"spar5-{iOML}" for iOML in range(1, 2 + 1)]
component_groups += [f"spar6-{iOML}" for iOML in range(1, 3 + 1)]
component_groups += [f"spar7-{iOML}" for iOML in range(1, 3 + 1)]
component_groups += [f"spar8-{iOML}" for iOML in range(1, 3 + 1)]
component_groups += [f"spar9-{iOML}" for iOML in range(1, 4 + 1)]
component_groups += [f"spar10-{iOML}" for iOML in range(1, 4 + 1)]
component_groups += [f"spar11-{iOML}" for iOML in range(1, 4 + 1)]
component_groups += [f"spar12-{iOML}" for iOML in range(1, 5 + 1)]
component_groups += [f"spar13-{iOML}" for iOML in range(1, 5 + 1)]
component_groups += [f"spar14-{iOML}" for iOML in range(1, 5 + 1)]
component_groups += [f"spar15-{iOML}" for iOML in range(1, 6 + 1)]
component_groups += [f"spar16-{iOML}" for iOML in range(1, 6 + 1)]
component_groups += [f"spar17-{iOML}" for iOML in range(1, 6 + 1)]
component_groups += [f"spar18-{iOML}" for iOML in range(1, 7 + 1)]
component_groups += [f"spar19-{iOML}" for iOML in range(1, 7 + 1)]
component_groups += [f"spar20-{iOML}" for iOML in range(1, 8 + 1)]
component_groups += [f"spar21-{iOML}" for iOML in range(1, 8 + 1)]
component_groups += [f"spar22-{iOML}" for iOML in range(1, 8 + 1)]
component_groups += [f"spar23-{iOML}" for iOML in range(1, 9 + 1)]
component_groups += [f"spar24-{iOML}" for iOML in range(1, 10 + 1)]
component_groups += [f"spar25-{iOML}" for iOML in range(1, 10 + 1)]
component_groups += [f"spar26-{iOML}" for iOML in range(1, 11 + 1)]
component_groups += [f"spar27-{iOML}" for iOML in range(1, 12 + 1)]
component_groups += [f"spar28-{iOML}" for iOML in range(1, 13 + 1)]
component_groups += [f"spar29-{iOML}" for iOML in range(1, 14 + 1)]
component_groups += [f"spar30-{iOML}" for iOML in range(1, 15 + 1)]
component_groups += [f"spar31-{iOML}" for iOML in range(1, 16 + 1)]
component_groups += [f"spar32-{iOML}" for iOML in range(1, 17 + 1)]
component_groups += [f"spar33-{iOML}" for iOML in range(1, 18 + 1)]
component_groups += [f"spar34-{iOML}" for iOML in range(1, 19 + 1)]
component_groups += [f"spar35-{iOML}" for iOML in range(1, 19 + 1)]
component_groups += [f"spar36-{iOML}" for iOML in range(1, 19 + 1)]
component_groups += [f"spar37-{iOML}" for iOML in range(1, 19 + 1)]
component_groups += [f"spar38-{iOML}" for iOML in range(1, 19 + 1)]
component_groups += [f"spar39-{iOML}" for iOML in range(1, 19 + 1)]
component_groups += [f"spar40-{iOML}" for iOML in range(1, 19 + 1)]

component_groups = sorted(component_groups)

# print(f"component group 89 = {component_groups[89]}")
# exit()

if args.useML:
    callback = gp_callback_generator(component_groups)

for icomp, comp in enumerate(component_groups):
    # caps2tacs.CompositeProperty.null(comp, null_material).register_to(tacs_model)

    # NOTE : need to make the struct DVs in TACS in the same order as the blade callback
    # which is done by components and then a local order

    # panel length variable
    Variable.structural(
        f"{comp}-" + TacsSteadyInterface.LENGTH_VAR, value=0.1
    ).set_bounds(
        lower=0.0,
        scale=1.0,
        state=True,  # need the length & width to be state variables
    ).register_to(
        wing
    )

    # stiffener pitch variable
    Variable.structural(f"{comp}-spitch", value=0.40).set_bounds(
        lower=0.05, upper=0.5, scale=1.0
    ).register_to(wing)

    # panel thickness variable, shortened DV name for ESP/CAPS, nastran requirement here
    Variable.structural(f"{comp}-pthick", value=0.1).set_bounds(
        lower=2e-3 if "OML" in comp else 1e-4, upper=0.2, scale=100.0
    ).register_to(wing)

    # stiffener height
    Variable.structural(f"{comp}-sheight", value=0.2).set_bounds(
        lower=2e-3 if "OML" in comp else 1e-4, upper=0.4, scale=10.0
    ).register_to(wing)

    # stiffener thickness
    Variable.structural(f"{comp}-sthick", value=0.1).set_bounds(
        lower=2e-3 if "OML" in comp else 1e-4, upper=0.4, scale=100.0
    ).register_to(wing)

    Variable.structural(
        f"{comp}-" + TacsSteadyInterface.WIDTH_VAR, value=0.1
    ).set_bounds(
        lower=0.0,
        scale=1.0,
        state=True,  # need the length & width to be state variables
    ).register_to(
        wing
    )


# register the wing body to the model
wing.register_to(f2f_model)

# INITIAL STRUCTURE MESH, SINCE NO STRUCT SHAPE VARS
# --------------------------------------------------

# tacs_aim.setup_aim()
# if args.newMesh:
#    tacs_aim.pre_analysis()

# SCENARIOS
# ----------------------------------------------------

# make a funtofem scenario
climb = Scenario.steady("climb_turb", steps=350, uncoupled_steps=200)  # 2000
Function.ksfailure(ks_weight=100.0, safety_factor=1.5).optimize(
    scale=1.0, upper=1.0, objective=False, plot=True, plot_name="ks-climb"
).register_to(climb)
Function.mass().optimize(
    scale=1.0e-4, objective=True, plot=True, plot_name="mass"
).register_to(climb)
climb.set_temperature(T_ref=216, T_inf=216)
climb.set_flow_ref_vals(qinf=3.16e4)
climb.register_to(f2f_model)

# NON-ADJACENCY COMPOSITE FUNCTIONS [not just variables involved]
# ---------------------------------------------------------------

# There are None for this optimization problem
# the adjacency constraints come at a later section near the optimizer..

# DISCIPLINE INTERFACES AND DRIVERS
# -----------------------------------------------------

solvers = SolverManager(comm)
# solvers.flow = Fun3dInterface(comm, f2f_model, fun3d_dir="meshes")
solvers.structural = TacsSteadyInterface.create_from_bdf(
    model=f2f_model,
    comm=comm,
    nprocs=args.procs,
    # bdf_file=tacs_aim.root_dat_file if args.newMesh else "struct/tacs.dat",
    # prefix=tacs_aim.root_analysis_dir if args.newMesh else "struct",
    bdf_file="struct/tacs.dat",
    prefix="struct",
    callback=callback,
    panel_length_dv_index=0,
    panel_width_dv_index=5,
)

f2f_model.print_memory_size(comm, root=0, starting_message="After solvers")


# read in aero loads
aero_loads_file = os.path.join(os.getcwd(), "cfd", "loads", "uncoupled_turb_loads.txt")

transfer_settings = TransferSettings(npts=200)

# build the shape driver from the file
tacs_driver = OnewayStructDriver.prime_loads_from_file(
    filename=aero_loads_file,
    solvers=solvers,
    model=f2f_model,
    nprocs=args.procs,
    transfer_settings=transfer_settings,
    init_transfer=True,
)

f2f_model.print_memory_size(comm, root=0, starting_message="After driver")

mem4 = process_memory()
# dmem3 = mem4 - mem3
dmem3 = mem4 - mem1
print(f"memory added during solvers, drivers = {dmem3} GB, total = {mem4} GB")
# exit()

# PYOPTSPARSE OPTMIZATION
# -------------------------------------------------------------

# create an OptimizationManager object for the pyoptsparse optimization problem
design_out_file = os.path.join(
    base_dir, "design", "ML-sizing.txt" if args.useML else "CF-sizing.txt"
)

# f2f_model.read_design_variables_file(comm, "design/state-sizing.txt")

opt_manager = OptimizationManager(
    tacs_driver,
    design_out_file=design_out_file,
    hot_start=hot_start,
    plot_hist=False,
    debug=True,
    sparse=True,
)

# create the pyoptsparse optimization problem
opt_problem = Optimization("hsctOpt", opt_manager.eval_functions)

# add funtofem model variables to pyoptsparse
opt_manager.register_to_problem(opt_problem)

# VARS_ONLY / SPARSE LINEAR COMPOSITE FUNCTIONS
# -------------------------------------------------------
# TBD, this will be a bit tricky here, prob just need some None checks
f2f_model.print_memory_size(comm, root=0, starting_message="Before composite functions")

# it's far more memory efficient to not register these ~13k composite functions below
# directly into the f2f_model [took up like 20 GB of data on my desktop computer]
# instead, it's more efficient to directly build a pyoptsparse sparse linear constraint object
# then delete the composite function and it only takes up like 2 GB!

# this was a 2 day investigation into python memory management. ended up realizing that
# even if I later delete the composite functions in the funtofem model and free up the space.
# python never gives back that RAM => which is kind of bizarre, but a TEDTalk on python memory confirmed this.
# so better to never allocate that much memory in the first place.. another option was to try and use
# __slots__ [tuple-based] instead of __dict__ [dictionary-based] version of compositeFunctions to store the memory [smaller memory
#  because tuple-based is immutable object]. However, better to never allocate that much memory if we don't need to in the first place.

# skin thickness adjacency constraints
variables = f2f_model.get_variables()
adjacency_scale = 10.0
thick_adj = 2.5e-3

nribs = 20
nspars = 40
nOML = nribs - 1

adj_types = ["pthick", "sthick", "sheight"]
adj_values = [2.5e-3, 2.5e-3, 10e-3]

adj_prefix_lists = []
ncomp = 0

# OMLtop, bot chordwise adjacency
for prefix in ["OMLtop", "OMLbot"]:
    # chordwise adjacency
    adj_prefix_lists += [
        [f"{prefix}{iOML}-{ispar}", f"{prefix}{iOML}-{ispar+1}"]
        for iOML in range(1, nOML + 1)
        for ispar in range(1, nspars + 1)
    ]
    # spanwise adjacency
    adj_prefix_lists += [
        [f"{prefix}{iOML}-{ispar}", f"{prefix}{iOML+1}-{ispar}"]
        for iOML in range(1, nOML)
        for ispar in range(1, nspars + 2)
    ]

# rib chordwise adjacency
adj_prefix_lists += [
    [f"rib{irib}-{ispar}", f"rib{irib}-{ispar+1}"]
    for irib in range(1, nribs + 1)
    for ispar in range(1, nspars + 1)
]

# spar adjacencies
spar_prefixes = [f"spar{ispar}" for ispar in range(1, nspars + 1)] + [
    "LEspar",
    "TEspar",
]
adj_prefix_lists += [
    [f"{spar_prefix}-{iOML}", f"{spar_prefix}-{iOML+1}"]
    for spar_prefix in spar_prefixes
    for iOML in range(1, nOML + 1)
]

# print(f"num adj prefix lists = {len(adj_prefix_lists)}", flush=True)
# exit()


def get_var_index(variable):
    for ivar, var in enumerate(f2f_model.get_variables(optim=True)):
        if var == variable:
            return ivar


n = len(adj_prefix_lists)
if comm.rank == 0:
    print(f"starting adj functions up to {3*len(adj_prefix_lists)}", flush=True)

nvariables = len(f2f_model.get_variables(optim=True))
# nvariables = 6456 # TODO : Fix this later.. [said it was 9684 instead of 6456 when I used len(...)]

for i, prefix_list in enumerate(adj_prefix_lists):
    left_prefix = prefix_list[0]
    right_prefix = prefix_list[1]
    if comm.rank == 0:
        print(f"adjacency funcs {ncomp}")
        if i < n - 1:
            print("\033[1A", end="")

    for iadj, adj_type in enumerate(adj_types):
        adj_value = adj_values[iadj]
        left_var = f2f_model.get_variables(f"{left_prefix}-{adj_type}")
        right_var = f2f_model.get_variables(f"{right_prefix}-{adj_type}")
        if left_var is not None and right_var is not None:
            adj_constr = left_var - right_var
            adj_constr.set_name(f"{left_prefix}-adj{i}_{adj_type}").optimize(
                lower=-adj_value, upper=adj_value, scale=10.0, objective=False
            ).setup_sparse_gradient(f2f_model)

            opt_manager.register_sparse_constraint(opt_problem, adj_constr)

            del adj_constr
            ncomp += 1

if comm.rank == 0:
    print(f"done with adjacency functons..")

prefix_lists = []

# OMLtop, bot
for _prefix in ["OMLtop", "OMLbot"]:
    prefix_lists += [
        f"{_prefix}{iOML}-{ispar}"
        for iOML in range(1, nOML + 1)
        for ispar in range(1, nspars + 2)
    ]

# ribs
prefix_lists += [
    f"rib{irib}-{ispar}"
    for irib in range(1, nribs + 1)
    for ispar in range(1, nspars + 2)
]
# spars
prefix_lists += [
    f"{spar_prefix}-{iOML}"
    for spar_prefix in spar_prefixes
    for iOML in range(1, nOML + 1)
]

n2 = len(prefix_lists)
for j, prefix in enumerate(prefix_lists):

    if comm.rank == 0:
        print(f"reg rel comp funcs {ncomp}")
        if j < n2 - 1:
            print("\033[1A", end="")

    skin_var = f2f_model.get_variables(f"{prefix}-pthick")
    sthick_var = f2f_model.get_variables(f"{prefix}-sthick")
    sheight_var = f2f_model.get_variables(f"{prefix}-sheight")
    spitch_var = f2f_model.get_variables(f"{prefix}-spitch")

    # stiffener - skin thickness adjacency here
    if skin_var is not None and sthick_var is not None:
        adj_value = thick_adj
        adj_constr = skin_var - sthick_var
        adj_constr.set_name(f"{prefix}-skin_stiff_T").optimize(
            lower=-adj_value, upper=adj_value, scale=10.0, objective=False
        ).setup_sparse_gradient(f2f_model)

        opt_manager.register_sparse_constraint(opt_problem, adj_constr)

        del adj_constr
        ncomp += 1

        # minimum stiffener spacing pitch > 2 * height
    if spitch_var is not None and sheight_var is not None:
        min_spacing_constr = spitch_var - 2 * sheight_var
        min_spacing_constr.set_name(f"{prefix}-sspacing").optimize(
            lower=0.0, scale=1.0, objective=False
        ).setup_sparse_gradient(f2f_model)

        opt_manager.register_sparse_constraint(opt_problem, min_spacing_constr)

        del min_spacing_constr
        ncomp += 1

    # minimum stiffener AR
    if sheight_var is not None and sthick_var is not None:
        min_stiff_AR = sheight_var - 2.0 * sthick_var
        min_stiff_AR.set_name(f"{prefix}-minstiffAR").optimize(
            lower=0.0, scale=1.0, objective=False
        ).setup_sparse_gradient(f2f_model)

        opt_manager.register_sparse_constraint(opt_problem, min_stiff_AR)

        del min_stiff_AR
        ncomp += 1

    # maximum stiffener AR (for regions with tensile strains where crippling constraint won't be active)
    if sheight_var is not None and sthick_var is not None:
        max_stiff_AR = sheight_var - 8.0 * sthick_var
        max_stiff_AR.set_name(f"{prefix}-maxstiffAR").optimize(
            upper=0.0, scale=1.0, objective=False
        ).setup_sparse_gradient(f2f_model)

        opt_manager.register_sparse_constraint(opt_problem, max_stiff_AR)

        del max_stiff_AR
        ncomp += 1

if comm.rank == 0:
    print(f"number of composite functions = {ncomp}", flush=True)

# return to Optimization
# -------------------------------------

mem5 = process_memory()
dmem = mem5 - mem4
print(
    f"memory added while registering to optimization problem (w/ cfunc clearing) = {dmem} GB, total = {mem5} GB"
)
# exit()

# this clear call is done inside the optimizer now..
f2f_model.clear_vars_only_composite_functions()
gc.collect()
# still need to cleanup the removed objects now..
f2f_model.print_memory_size(
    comm, root=0, starting_message="After optimizer, before running"
)
mem6 = process_memory()
dmem = mem6 - mem5
print(
    f"memory added while registering to optimization problem (w/ cfunc clearing) = {dmem} GB, total = {mem6} GB"
)
print("Did it actually clear out that memory for composite functions?")
# exit()

# run an SNOPT optimization

design_folder = os.path.join(base_dir, "design")
if not os.path.exists(design_folder):
    os.mkdir(design_folder)
history_file = os.path.join(
    design_folder, "ML-sizing.hst" if args.useML else "CF-sizing.hst"
)
store_history_file = history_file if store_history else None
hot_start_file = history_file if hot_start else None

snoptimizer = SNOPT(
    options={
        "Print frequency": 1000,
        "Summary frequency": 10000000,
        "Major feasibility tolerance": 1e-6,
        "Major optimality tolerance": 1e-6,
        "Verify level": 0,
        "Major iterations limit": 15000,
        "Minor iterations limit": 150000000,
        "Iterations limit": 100000000,
        "Major step limit": 5e-2,  # had this off I think (but this maybe could be on)
        "Nonderivative linesearch": True,  # turns off derivative linesearch
        "Linesearch tolerance": 0.9,
        "Difference interval": 1e-6,
        "Function precision": 1e-10,
        "New superbasics limit": 2000,
        "Penalty parameter": 1.0,  # had this off for faster opt in the single panel case
        # however ksfailure becomes too large with this off. W/ on merit function goes down too slowly though
        # try intermediate value btw 0 and 1 (smaller penalty)
        # this may be the most important switch to change for opt performance w/ ksfailure in the opt
        "Scale option": 1,
        "Hessian updates": 40,
    }
)

sol = snoptimizer(
    opt_problem,
    sens=opt_manager.eval_gradients,
    storeHistory=history_file,
    hotStart=history_file if args.hotstart else None,
)

sol_xdict = sol.xStar
if comm.rank == 0:
    print(f"Final solution = {sol_xdict}", flush=True)
