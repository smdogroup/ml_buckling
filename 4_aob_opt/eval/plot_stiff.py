import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import argparse

# import openmdao.api as om
from funtofem import *
from mpi4py import MPI
from tacs import caps2tacs
import os, sys

sys.path.append("../")

parent_parser = argparse.ArgumentParser(add_help=False)
parent_parser.add_argument("--useML", type=bool, default=False)
args = parent_parser.parse_args()

if args.useML:
    from _gp_metal__callback import gp_callback_generator

    model_name = "ML-oneway"
else:
    from _closed_form_metal_callback import closed_form_callback as callback

    model_name = "CF-oneway"


comm = MPI.COMM_WORLD

base_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(base_dir, "..")
csm_path = os.path.join(base_dir, "geometry", "gbm.csm")

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
component_groups = [f"uOML{iOML}" for iOML in range(1, nOML + 1)]
component_groups += [f"rib{irib}" for irib in range(1, nribs + 1)]
for prefix in ["spLE", "spTE"]:
    component_groups += [f"{prefix}{iOML}" for iOML in range(1, nOML + 1)]
component_groups += [f"lOML{iOML}" for iOML in range(1, nOML + 1)]

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
    sheight_var = (
        Variable.structural(f"{comp}-sheight", value=0.05)
        .set_bounds(lower=0.002, upper=0.1, scale=10.0)
        .register_to(wing)
    )

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

# register the wing body to the model
wing.register_to(f2f_model)

# make a funtofem scenario
cruise = Scenario.steady("climb_turb", steps=2)  # 2000
# increase ksfailure scale to make it stricter on infeasibility for that.
Function.ksfailure(ks_weight=20.0, safety_factor=1.5).optimize(
    scale=1.0, upper=1.0, objective=False, plot=True, plot_name="ks-cruise"
).register_to(cruise)

Function.mass().optimize(
    scale=1.0e-3, objective=True, plot=True, plot_name="mass"
).register_to(cruise)
cruise.register_to(f2f_model)

# read the DV file
design_out_file = os.path.join(
    base_dir, "design", "ML-metal-sizing.txt" if args.useML else "CF-metal-sizing.txt"
)

# reload previous design
f2f_model.read_design_variables_file(comm, design_out_file)


# plot the OML stiffeners

# Create a simple plot
fig, ax = plt.subplots()
ax.plot([1, 1, 1], [1, 1, 1], color="white")

# make the wing upper skin panel rectangle
sspan = 14.0
wthick = 0.04
ax.add_patch(Rectangle((0, 0), sspan, wthick, color="gray"))

# Add a rectangle for each stiffener
# ax.add_patch(Rectangle((2, 2), 1, 3, color="gray"))

# scale for stiffeners in the plot
scale = 7.0

for iOML in range(1, nOML + 1):
    if args.useML:
        cen_pos = sspan * iOML / (nOML + 1)
    else:
        cen_pos = sspan * (nOML + 1 - iOML) / (nOML + 1)

    # get stiffener height, thick
    sheight = f2f_model.get_variables(f"uOML{iOML}-sheight").value
    sthick = f2f_model.get_variables(f"uOML{iOML}-sthick").value

    left_pos = (cen_pos - sthick / 2.0 * scale, wthick)
    ax.add_patch(Rectangle(left_pos, sthick * scale, sheight * scale, color="gray"))

plt.xlabel("X-AXIS")
plt.ylabel("Y-AXIS")
plt.title("PLOT-1")
plt.axis("equal")
# plt.show()
plt.savefig("CF-stiff.png" if not (args.useML) else "ML-stiff.png", dpi=400)
