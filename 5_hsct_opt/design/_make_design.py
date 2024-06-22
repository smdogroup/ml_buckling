"""
This is the fully coupled aerothermoelastic optimization of the HSCT.
NOTE: You need to run the 1_sizing_optimization.py first and leave the
optimal panel thickness design variables in the meshes folder before running this.

NOTE : don't call this script with mpiexec_mpt, call it with python (otherwise system calls won't work)
"""

import os
from mpi4py import MPI
from funtofem import *
from tacs import caps2tacs

comm = MPI.COMM_WORLD
base_dir = os.path.dirname(os.path.abspath(__file__))
csm_path = os.path.join(base_dir, "..", "geometry", "hsct.csm")

# F2F MODEL and SHAPE MODELS
# ----------------------------------------

f2f_model = FUNtoFEMmodel("hsct-sizing")
tacs_model = caps2tacs.TacsModel.build(
    csm_file=csm_path,
    comm=comm,
    problem_name="structDesign",
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
tacs_aim.set_config_parameter("wing:allOMLgroups", 0)
tacs_aim.set_config_parameter("wing:includeTE", 0)

for proc in tacs_aim.active_procs:
    if comm.rank == proc:
        aim = tacs_model.mesh_aim.aim
        aim.input.Mesh_Sizing = {
            "chord": {"numEdgePoints": 20},
            "span": {"numEdgePoints": 8},
            "vert": {"numEdgePoints": 4},
            "LEribFace": {"tessParams": [0.03, 0.1, 3]},
            "LEribEdge": {"numEdgePoints": 20},
        }

# add tacs constraints in
caps2tacs.PinConstraint("root").register_to(tacs_model)
caps2tacs.PinConstraint("station2").register_to(tacs_model)
caps2tacs.TemperatureConstraint("midplane", temperature=0).register_to(tacs_model)

# BODIES AND STRUCT DVs
# -------------------------------------------------

wing = Body.aerothermoelastic("wing", boundary=5)
# aerothermoelastic

# setup the material and shell properties
titanium_alloy = caps2tacs.Isotropic.titanium_alloy().register_to(tacs_model)

nribs = int(tacs_model.get_config_parameter("wing:nribs"))
nspars = int(tacs_model.get_config_parameter("wing:nspars"))
nOML = nribs - 1

for irib in range(1, nribs + 1):
    name = f"rib{irib}"
    prop = caps2tacs.ShellProperty(
        caps_group=name, material=titanium_alloy, membrane_thickness=0.04
    ).register_to(tacs_model)
    Variable.structural(name, value=0.01).set_bounds(
        lower=0.001, upper=0.15, scale=100.0
    ).register_to(wing)

for ispar in range(1, nspars + 1):
    name = f"spar{ispar}"
    prop = caps2tacs.ShellProperty(
        caps_group=name, material=titanium_alloy, membrane_thickness=0.04
    ).register_to(tacs_model)
    Variable.structural(name, value=0.01).set_bounds(
        lower=0.001, upper=0.15, scale=100.0
    ).register_to(wing)

for iOML in range(1, nOML + 1):
    name = f"OML{iOML}"
    prop = caps2tacs.ShellProperty(
        caps_group=name, material=titanium_alloy, membrane_thickness=0.04
    ).register_to(tacs_model)
    Variable.structural(name, value=0.01).set_bounds(
        lower=0.001, upper=0.15, scale=100.0
    ).register_to(wing)

    name = f"LE{iOML}"
    prop = caps2tacs.ShellProperty(
        caps_group=name, material=titanium_alloy, membrane_thickness=0.04
    ).register_to(tacs_model)
    Variable.structural(name, value=0.01).set_bounds(
        lower=0.001, upper=0.15, scale=100.0
    ).register_to(wing)

    # name = f"TE{iOML}"
    # prop = caps2tacs.ShellProperty(
    #     caps_group=name, material=titanium_alloy, membrane_thickness=0.04
    # ).register_to(tacs_model)
    # Variable.structural(name, value=0.01).set_bounds(
    #     lower=0.001, upper=0.15, scale=100.0
    # ).register_to(wing)

for prefix in ["LE", "TE"]:
    name = f"{prefix}spar"
    prop = caps2tacs.ShellProperty(
        caps_group=name, material=titanium_alloy, membrane_thickness=0.04
    ).register_to(tacs_model)
    Variable.structural(name, value=0.01).set_bounds(
        lower=0.001, upper=0.15, scale=100.0
    ).register_to(wing)

# structural shape variables
rib_a1 = (
    Variable.shape(f"wing:rib_a1", value=1.0)
    .set_bounds(lower=0.65, upper=1.35)
    .register_to(wing)
)
rib_a2 = (
    Variable.shape(f"wing:rib_a2", value=0.0)
    .set_bounds(lower=-0.3, upper=0.3)
    .register_to(wing)
)
spar_a1 = (
    Variable.shape(f"wing:spar_a1", value=1.0)
    .set_bounds(lower=0.65, upper=1.35)
    .register_to(wing)
)
spar_a2 = (
    Variable.shape(f"wing:spar_a2", value=0.0)
    .set_bounds(lower=-0.3, upper=0.3)
    .register_to(wing)
)

wing_c1 = (
    Variable.shape("wing:c1", value=48.0)
    .set_bounds(lower=1.0, upper=1.4)
    .register_to(wing)
)

wing.register_to(f2f_model)
# end of wing section

# create an OptimizationManager object for the pyoptsparse optimization problem
design_out_file = os.path.join(base_dir, "sizing.txt")  # internal-struct.txt

# reload previous design
f2f_model.read_design_variables_file(comm, design_out_file)

# make input dict from the thickness variables
input_dict = {
    var.name: var.value
    for var in f2f_model.get_variables()
    if var.analysis_type in ["structural", "shape"]
}

tacs_model.update_design(input_dict)
tacs_model.setup(include_aim=True)

# animate caps2tacs variables
shape_var_dict = {wing_c1: [48.0]}
tacs_model.animate_shape_vars(shape_var_dict)
