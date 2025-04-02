"""
Sean P. Engelstad, Georgia Tech 2023

Local machine optimization for the panel thicknesses using nribs-1 OML panels and nribs-1 LE panels
"""

from funtofem import *
import time
import numpy as np
import time
start_time = time.time()
import argparse

parent_parser = argparse.ArgumentParser(add_help=False)
parent_parser.add_argument("--spanMult", type=float, default=1.0) # recommend 1.0, 1.1, 1.2, 1.3
args = parent_parser.parse_args()

# import openmdao.api as om
from mpi4py import MPI
from tacs import caps2tacs, pytacs
import os


comm = MPI.COMM_WORLD

base_dir = os.path.dirname(os.path.abspath(__file__))
csm_path = os.path.join(base_dir, "geometry", "gbm.csm")

# F2F MODEL and SHAPE MODELS
# ----------------------------------------

f2f_model = FUNtoFEMmodel('model')
tacs_model = caps2tacs.TacsModel.build(
    csm_file=csm_path,
    comm=comm,
    problem_name="capsStruct0",
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
# tacs_aim.set_design_parameter("spanMult", args.spanMult)
#tacs_aim.set_config_parameter("fullRootRib", 0)

egads_aim = tacs_model.mesh_aim

# 17000 if elem_mult = 1
# 70000 if elem_mult = 2
elem_mult = 2

if comm.rank == 0:
    aim = egads_aim.aim
    aim.input.Mesh_Sizing = {
        "chord": {"numEdgePoints": 20*elem_mult},
        "span": {"numEdgePoints": 10*elem_mult},
        "vert": {"numEdgePoints": 10*elem_mult},
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

# caps2tacs.PinConstraint("root", dof_constraint=12346).register_to(tacs_model)
caps2tacs.PinConstraint("root", dof_constraint=246).register_to(tacs_model)
caps2tacs.PinConstraint("sob", dof_constraint=13).register_to(tacs_model)

# register the wing body to the model
wing.register_to(f2f_model)

# INITIAL STRUCTURE MESH, SINCE NO STRUCT SHAPE VARS
# --------------------------------------------------

tacs_aim.setup_aim()
tacs_aim.pre_analysis()

# move the file to the right dir
import os
if comm.rank == 0:
    if not os.path.exists("design/struct"):
        os.mkdir("design/struct")

if comm.rank == 0:
    os.system('mv capsStruct0_0/Scratch/tacs/tacs.bdf design/struct')
    os.system('mv capsStruct0_0/Scratch/tacs/tacs.dat design/struct')
