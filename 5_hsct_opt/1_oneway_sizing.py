"""
Sean P. Engelstad, Georgia Tech 2023

Local machine optimization for the panel thicknesses using all OML and LE panels (more design variables) but no shape variables
"""

from funtofem import *
from pyoptsparse import SNOPT, Optimization

# script inputs
hot_start = False
store_history = True

# import openmdao.api as om
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

f2f_model = FUNtoFEMmodel(model_name)
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
tacs_aim.set_config_parameter("wing:nspars", 16)

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

# old way was to explicitly define the panels by looking at geom
# spar_npanels_dict = {
#     1: 2, 2: 4, 3: 5, 4: 7, 5: 9,
#     6: 12, 7: 17, 8: 19, 
# }
# OML_npanels_dict = {
#     0: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 4,
#     7: 4, 8: 5, 9: 5, 10: 6, 11: 6,  12: 6, 13: 7,
#     14: 7, 15: 7, 16: 7, 17: 7, 18: 8, 19: 8, 20:8
# }

# component_groups = []

# for irib in range(1, nribs+1):
#     istart = OML_npanels_dict[irib]
#     for ichord in range(istart, nspars+2):
#         component_groups += [f"rib{irib}-{ichord}"]

# for iOML in range(1,nOML+1):
#     istart = OML_npanels_dict[iOML]
#     for ichord in range(istart,nspars+2):
#         component_groups += [f"OMLtop{iOML}-{ichord}", f"OMLbot{iOML}-{ichord}"]

# for ispar in range(1, nspars+1):
#     ifinal = spar_npanels_dict[ispar]
#     for ispan in range(1, ifinal+1):
#         component_groups += [f"spar{ispar}-{ispan}"]

# for prefix in ["LEspar", "TEspar"]:
#     for ispan in range(1, nOML+1):
#         component_groups += [f"{prefix}-{ispan}"]

# print(f"component groups = {component_groups}")
# exit()

# new way is to use ESP/CAPS display attributes tool to get all capsGroup attributes
# (namely use displayFilter: capsGroup then ? entry)
# store it in a string and strtok the string. 
caps_component_groups_str = "spar14-18, OMLtop18-15, OMLtop18-14, spar14-17, rib18-15, rib18-14, spar14-19, rib19-15, rib19-14, spar15-18, OMLtop18-16, OMLtop19-15, OMLtop17-15, OMLtop19-14, LEspar-18, OMLtop17-14, rib17-14, spar14-16, rib17-15, spar15-17, rib18-16, LEspar-17, OMLbot18-15, OMLbot18-14, LEspar-19, spar15-19, rib19-16, TEspar-18, OMLtop19-16, OMLtop17-16, spar16-18, OMLtop18-17, rib20-15, OMLtop16-15, spar13-17, OMLtop17-13, OMLtop16-14, spar13-16, rib17-13, rib16-14, spar14-15, rib16-15, spar15-16, rib17-16, OMLbot17-15, OMLbot17-14, spar16-17, rib18-17, OMLbot19-15, OMLbot18-16, OMLbot19-14, rib20-16, TEspar-19, OMLbot19-16, OMLtop17-17, OMLtop16-16, OMLbot18-17, OMLtop15-15, OMLtop16-13, OMLtop15-14, spar13-15, rib16-13, LEspar-16, OMLbot16-14, rib15-14, spar14-14, rib15-15, spar15-15, rib16-16, OMLbot16-15, spar16-16, rib17-17, OMLbot17-16, OMLbot17-13, TEspar-17, OMLbot17-17, OMLtop16-17, OMLtop15-16, OMLtop14-15, OMLtop15-13, OMLtop14-14, spar13-14, rib15-13, LEspar-15, OMLbot16-13, OMLbot15-14, rib14-14, spar14-13, rib14-15, spar15-14, rib15-16, OMLbot15-15, spar16-15, rib16-17, OMLbot16-16, TEspar-16, OMLbot16-17, OMLtop15-17, OMLtop14-16, OMLtop13-15, OMLtop14-13, OMLtop13-14, rib14-13, spar13-13, LEspar-14, OMLbot15-13, OMLbot14-14, rib13-14, spar14-12, rib13-15, spar15-13, rib14-16, OMLbot14-15, spar16-14, rib15-17, OMLbot15-16, TEspar-15, OMLbot15-17, OMLtop14-17, OMLtop13-16, OMLtop12-15, spar12-14, OMLtop14-12, OMLtop13-13, OMLtop12-14, spar12-13, rib14-12, rib13-13, spar13-12, OMLbot14-13, OMLbot13-14, rib12-14, spar14-11, rib12-15, spar15-12, rib13-16, OMLbot13-15, spar16-13, rib14-17, OMLbot14-16, TEspar-14, OMLbot14-17, OMLtop13-17, OMLtop12-16, OMLtop11-15, OMLtop13-12, OMLtop12-13, OMLtop11-14, spar12-12, rib13-12, LEspar-13, OMLbot13-13, rib12-13, spar13-11, OMLbot14-12, OMLbot12-14, rib11-14, spar14-10, rib11-15, spar15-11, rib12-16, OMLbot12-15, spar16-12, rib13-17, OMLbot13-16, TEspar-13, OMLbot13-17, OMLtop12-17, OMLtop11-16, OMLtop10-15, OMLtop12-12, OMLtop11-13, OMLtop10-14, rib12-12, spar12-11, LEspar-12, OMLbot13-12, OMLbot12-13, rib11-13, spar13-10, OMLbot11-14, rib10-14, spar14-9, rib10-15, spar15-10, rib11-16, OMLbot11-15, spar16-11, rib12-17, OMLbot12-16, TEspar-12, OMLbot12-17, OMLtop11-17, OMLtop10-16, OMLtop9-15, spar11-12, OMLtop12-11, OMLtop11-12, OMLtop10-13, OMLtop9-14, spar11-11, rib12-11, rib11-12, spar12-10, OMLbot12-12, OMLbot11-13, rib10-13, spar13-9, OMLbot10-14, rib9-14, spar14-8, rib9-15, spar15-9, rib10-16, OMLbot10-15, spar16-10, rib11-17, OMLbot11-16, TEspar-11, OMLbot11-17, OMLtop10-17, OMLtop9-16, OMLtop8-15, OMLtop11-11, OMLtop10-12, OMLtop9-13, OMLtop8-14, spar11-10, rib11-11, LEspar-11, OMLbot11-12, rib10-12, spar12-9, OMLbot12-11, OMLbot10-13, rib9-13, spar13-8, OMLbot9-14, rib8-14, spar14-7, rib8-15, spar15-8, rib9-16, OMLbot9-15, spar16-9, rib10-17, OMLbot10-16, TEspar-10, OMLbot10-17, OMLtop9-17, OMLtop8-16, OMLtop7-15, OMLtop10-11, OMLtop9-12, OMLtop8-13, OMLtop7-14, rib10-11, spar11-9, LEspar-10, OMLbot11-11, OMLbot10-12, rib9-12, spar12-8, OMLbot9-13, rib8-13, spar13-7, OMLbot8-14, rib7-14, spar14-6, rib7-15, spar15-7, rib8-16, OMLbot8-15, spar16-8, rib9-17, OMLbot9-16, TEspar-9, OMLbot9-17, OMLtop8-17, OMLtop7-16, OMLtop6-15, spar10-10, OMLtop10-10, OMLtop9-11, OMLtop8-12, OMLtop7-13, OMLtop6-14, spar10-9, rib10-10, rib9-11, spar11-8, OMLbot10-11, OMLbot9-12, rib8-12, spar12-7, OMLbot8-13, rib7-13, spar13-6, OMLbot7-14, rib6-14, spar14-5, rib6-15, spar15-6, rib7-16, OMLbot7-15, spar16-7, rib8-17, OMLbot8-16, TEspar-8, OMLbot8-17, OMLtop7-17, OMLtop6-16, OMLtop5-15, OMLtop9-10, OMLtop8-11, OMLtop7-12, OMLtop6-13, OMLtop5-14, spar10-8, rib9-10, LEspar-9, OMLbot9-11, rib8-11, spar11-7, OMLbot10-10, OMLbot8-12, rib7-12, spar12-6, OMLbot7-13, rib6-13, spar13-5, OMLbot6-14, rib5-14, spar14-4, rib5-15, spar15-5, rib6-16, OMLbot6-15, spar16-6, rib7-17, OMLbot7-16, TEspar-7, OMLbot7-17, OMLtop6-17, OMLtop5-16, OMLtop4-15, OMLtop8-10, OMLtop7-11, OMLtop6-12, OMLtop5-13, OMLtop4-14, rib8-10, spar10-7, LEspar-8, OMLbot9-10, OMLbot8-11, rib7-11, spar11-6, OMLbot7-12, rib6-12, spar12-5, OMLbot6-13, rib5-13, spar13-4, OMLbot5-14, rib4-14, spar14-3, rib4-15, spar15-4, rib5-16, OMLbot5-15, spar16-5, rib6-17, OMLbot6-16, TEspar-6, OMLbot6-17, OMLtop5-17, OMLtop4-16, OMLtop3-15, spar9-8, OMLtop8-9, OMLtop7-10, OMLtop6-11, OMLtop5-12, OMLtop4-13, OMLtop3-14, spar9-7, rib8-9, rib7-10, spar10-6, OMLbot8-10, OMLbot7-11, rib6-11, spar11-5, OMLbot6-12, rib5-12, spar12-4, OMLbot5-13, rib4-13, spar13-3, OMLbot4-14, rib3-14, spar14-2, rib3-15, spar15-3, rib4-16, OMLbot4-15, spar16-4, rib5-17, OMLbot5-16, TEspar-5, OMLbot5-17, OMLtop4-17, OMLtop3-16, OMLtop2-15, OMLbot8-9, OMLtop7-9, OMLtop6-10, OMLtop5-11, OMLtop4-12, OMLtop3-13, OMLtop2-14, rib7-9, spar9-6, LEspar-7, OMLbot7-9, OMLbot7-10, rib6-10, spar10-5, OMLbot6-11, rib5-11, spar11-4, OMLbot5-12, rib4-12, spar12-3, OMLbot4-13, rib3-13, spar13-2, OMLbot3-14, rib2-14, spar14-1, rib2-15, spar15-2, rib3-16, OMLbot3-15, spar16-3, rib4-17, OMLbot4-16, TEspar-4, OMLbot4-17, OMLtop3-17, OMLtop2-16, OMLtop1-15, spar8-7, OMLtop7-8, OMLtop6-9, OMLtop5-10, OMLtop4-11, OMLtop3-12, OMLtop2-13, OMLtop1-14, spar8-6, rib7-8, rib6-9, spar9-5, OMLbot6-9, OMLbot7-8, OMLbot6-10, rib5-10, spar10-4, OMLbot5-11, rib4-11, spar11-3, OMLbot4-12, rib3-12, spar12-2, OMLbot3-13, rib2-13, spar13-1, OMLbot2-14, rib1-14, rib1-15, spar15-1, rib2-16, OMLbot2-15, spar16-2, rib3-17, OMLbot3-16, TEspar-3, OMLbot3-17, OMLtop2-17, OMLtop1-16, OMLtop6-8, OMLtop5-9, OMLtop4-10, OMLtop3-11, OMLtop2-12, OMLtop1-13, rib6-8, spar8-5, LEspar-6, OMLbot6-8, rib5-9, spar9-4, OMLbot5-9, OMLbot5-10, rib4-10, spar10-3, OMLbot4-11, rib3-11, spar11-2, OMLbot3-12, rib2-12, spar12-1, OMLbot2-13, rib1-13, OMLbot1-14, rib1-16, OMLbot1-15, spar16-1, rib2-17, OMLbot2-16, TEspar-2, OMLbot2-17, OMLtop1-17, spar7-6, OMLtop6-7, OMLtop5-8, OMLtop4-9, OMLtop3-10, OMLtop2-11, OMLtop1-12, spar7-5, rib6-7, rib5-8, spar8-4, OMLbot5-8, OMLbot6-7, rib4-9, spar9-3, OMLbot4-9, OMLbot4-10, rib3-10, spar10-2, OMLbot3-11, rib2-11, spar11-1, OMLbot2-12, rib1-12, OMLbot1-13, OMLbot1-16, rib1-17, TEspar-1, OMLbot1-17, OMLtop5-7, OMLtop4-8, OMLtop3-9, OMLtop2-10, OMLtop1-11, rib5-7, spar7-4, LEspar-5, OMLbot5-7, rib4-8, spar8-3, OMLbot4-8, rib3-9, spar9-2, OMLbot3-9, OMLbot3-10, rib2-10, spar10-1, OMLbot2-11, rib1-11, OMLbot1-12, spar6-5, OMLtop5-6, OMLtop4-7, OMLtop3-8, OMLtop2-9, OMLtop1-10, rib5-6, spar6-4, rib4-7, spar7-3, OMLbot4-7, OMLbot5-6, rib3-8, spar8-2, OMLbot3-8, rib2-9, spar9-1, OMLbot2-9, OMLbot2-10, rib1-10, OMLbot1-11, spar5-5, OMLtop5-5, OMLtop4-6, OMLtop3-7, OMLtop2-8, OMLtop1-9, spar5-4, rib5-5, rib4-6, spar6-3, rib3-7, spar7-2, OMLbot4-6, OMLbot3-7, OMLbot5-5, rib2-8, spar8-1, OMLbot2-8, rib1-9, OMLbot1-9, OMLbot1-10, OMLtop4-5, OMLtop3-6, OMLtop2-7, OMLtop1-8, rib4-5, spar5-3, LEspar-4, OMLbot4-5, rib3-6, spar6-2, rib2-7, spar7-1, OMLbot3-6, OMLbot2-7, rib1-8, OMLbot1-8, spar4-4, OMLtop4-4, OMLtop3-5, OMLtop2-6, OMLtop1-7, spar4-3, rib4-4, rib3-5, spar5-2, OMLbot3-5, OMLbot4-4, rib2-6, spar6-1, rib1-7, OMLbot2-6, OMLbot1-7, OMLtop3-4, OMLtop2-5, OMLtop1-6, rib3-4, spar4-2, LEspar-3, OMLbot3-4, rib2-5, spar5-1, OMLbot2-5, rib1-6, OMLbot1-6, spar3-3, OMLtop3-3, OMLtop2-4, OMLtop1-5, spar3-2, rib3-3, rib2-4, spar4-1, OMLbot2-4, OMLbot3-3, rib1-5, OMLbot1-5, OMLtop2-3, OMLtop1-4, rib2-3, spar3-1, LEspar-2, rib1-4, OMLbot2-3, OMLbot1-4, spar2-2, OMLtop2-2, OMLtop1-3, spar2-1, rib2-2, rib1-3, OMLbot1-3, OMLbot2-2, OMLtop1-2, rib1-2, LEspar-1, OMLbot1-2, spar1-1, OMLtop1-1, rib1-1, OMLbot1-1"
component_groups = caps_component_groups_str.split(", ")
component_groups = sorted(component_groups)

#print(f"component group 89 = {component_groups[89]}")
#exit()

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

# PYOPTSPARSE OPTMIZATION
# -------------------------------------------------------------

# create an OptimizationManager object for the pyoptsparse optimization problem
design_out_file = os.path.join(base_dir, "design", "ML-sizing.txt" if args.useML else "CF-sizing.txt")

manager = OptimizationManager(
    tacs_driver, design_out_file=design_out_file, hot_start=hot_start, debug=True, sparse=True
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
