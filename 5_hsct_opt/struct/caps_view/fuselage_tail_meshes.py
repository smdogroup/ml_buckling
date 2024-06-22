"""
Sean P. Engelstad, Georgia Tech 2023

build the exploded meshes for the structural analysis for paraview
"""

# script inputs
hot_start = True
store_history = True
nprocs = 2

# import openmdao.api as om
from funtofem import *
from mpi4py import MPI
from tacs import caps2tacs
import os

comm = MPI.COMM_WORLD

base_dir = os.path.dirname(os.path.abspath(__file__))
csm_path = os.path.join(base_dir, "..", "..", "geometry", "hsct.csm")

# F2F MODEL and SHAPE MODELS
# ----------------------------------------

f2f_model = FUNtoFEMmodel("hsct-sizing")
tacs_model = caps2tacs.TacsModel.build(
    csm_file=csm_path,
    comm=comm,
    problem_name="_fuselageTail",
    active_procs=[_ for _ in range(nprocs)],
    verbosity=1,
)
tacs_model.mesh_aim.set_mesh(  # need a refined-enough mesh for the derivative test to pass
    edge_pt_min=100,
    edge_pt_max=200,
    global_mesh_size=0.003,
    max_surf_offset=0.05,
    max_dihedral_angle=5,
).register_to(
    tacs_model
)
f2f_model.structural = tacs_model

# run this with 3 procs in mpi and this should work!
#0 (off), 1 (upperOML), 2 (internalStruc), 3(lowerOML)
exploded_view = comm.rank + 1 
#exploded_view = 2

tacs_aim = tacs_model.tacs_aim
tacs_aim.set_config_parameter("mode:flow", 0)
tacs_aim.set_config_parameter("mode:struct", 1)

tacs_aim.set_config_parameter("view:wing", 0)
# set each value on a separate aim (not in parallel)
if comm.rank == 0: # fuselage
    tacs_aim.geometry.cfgpmtr["view:fuselage"].value = 1
    tacs_aim.geometry.cfgpmtr["view:tail"].value = 0
if comm.rank == 1: # fuselage
    tacs_aim.geometry.cfgpmtr["view:fuselage"].value = 0
    tacs_aim.geometry.cfgpmtr["view:tail"].value = 1

# Mesh_Sizing_dict = {}
# if exploded_view in [0,2]:
#     Mesh_Sizing_dict["chord"] = {"numEdgePoints": 20}
#     Mesh_Sizing_dict["vert"] = {"numEdgePoints": 4}
#     #Mesh_Sizing_dict["LEribFace"] = {"tessParams": [0.03, 0.1, 3]}
#     #Mesh_Sizing_dict["LEribEdge"] = {"numEdgePoints": 20}
# if exploded_view in [0,1,3]:
#     Mesh_Sizing_dict["chord"] = {"numEdgePoints": 20}
#     Mesh_Sizing_dict["span"] = {"numEdgePoints": 8}
#     #Mesh_Sizing_dict["LEribEdge"] = {"numEdgePoints": 20}
# print(f"mesh sizing dict = {Mesh_Sizing_dict}")

aim = tacs_model.mesh_aim.aim
# aim.input.Mesh_Sizing = Mesh_Sizing_dict        

# add tacs constraints in
caps2tacs.PinConstraint("dummy").register_to(tacs_model)

# BODIES AND STRUCT DVs
# -------------------------------------------------

wing = Body.aerothermoelastic("wing", boundary=5)
# aerothermoelastic

# setup the material and shell properties
titanium_alloy = caps2tacs.Isotropic.titanium_alloy().register_to(tacs_model)

if comm.rank == 0:
    name = "fuselageOML"
elif comm.rank == 1:
    name = "tailOML"
prop = caps2tacs.ShellProperty(
    caps_group=name, material=titanium_alloy, membrane_thickness=0.002
).register_to(tacs_model)
Variable.structural(name, value=0.01).set_bounds(
    lower=0.001, upper=0.002, scale=100.0
).register_to(wing)


# register the wing body to the model
wing.register_to(f2f_model)

# INITIAL STRUCTURE MESH, SINCE NO STRUCT SHAPE VARS
# --------------------------------------------------
tacs_aim.setup_aim()
tacs_aim.pre_analysis()