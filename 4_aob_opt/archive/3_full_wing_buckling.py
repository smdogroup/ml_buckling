"""
Sean P. Engelstad, Georgia Tech 2024

Goal here is to determine whether the global, local buckling constraints were conservative to the full wing bucklng modes.
"""

from funtofem import *
import time
from pyoptsparse import SNOPT, Optimization
import numpy as np
import argparse
import time
from tacs import pytacs, problems
start_time = time.time()

# import openmdao.api as om
from mpi4py import MPI
from tacs import caps2tacs
import os


parent_parser = argparse.ArgumentParser(add_help=False)
parent_parser.add_argument("--procs", type=int, default=6)
parent_parser.add_argument(
    "--hotstart", default=False, action=argparse.BooleanOptionalAction
)
parent_parser.add_argument(
    "--useML", default=False, action=argparse.BooleanOptionalAction
)
parent_parser.add_argument(
    "--metal", default=True, action=argparse.BooleanOptionalAction
)
args = parent_parser.parse_args()

if args.useML and args.metal:
    from _gp_metal_callback import gp_callback_generator

    model_name = "ML-metal-oneway"

elif args.useML and not args.metal:
    from _gp_callback import gp_callback_generator

    model_name = "ML-oneway"

elif not args.useML and args.metal:
    from _closed_form_metal_callback import closed_form_callback as callback

    model_name = "CF-metal-oneway"

else:
    from _closed_form_callback import closed_form_callback as callback

    model_name = "CF-oneway"


comm = MPI.COMM_WORLD

base_dir = os.path.dirname(os.path.abspath(__file__))
csm_path = os.path.join(base_dir, "geometry", "gbm.csm")

# F2F MODEL and SHAPE MODELS
# ----------------------------------------

f2f_model = FUNtoFEMmodel(model_name)
tacs_model = caps2tacs.TacsModel.build(
    csm_file=csm_path,
    comm=comm,
    problem_name="capsBuckle",
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

# make a funtofem scenario
climb = Scenario.steady("climb_turb", steps=2)  # 2000
# increase ksfailure scale to make it stricter on infeasibility for that.
Function.ksfailure(ks_weight=20.0, safety_factor=1.5).optimize(
    scale=1.0, upper=1.0, objective=False, plot=True, plot_name="ks-climb"
).register_to(climb)

Function.mass().optimize(
    scale=1.0e-3, objective=True, plot=True, plot_name="mass"
).register_to(climb)
climb.register_to(f2f_model)

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

transfer_settings = TransferSettings(npts=50, beta=0.1)

# build the shape driver from the file
tacs_driver = OnewayStructDriver.prime_loads_from_file(
    filename=aero_loads_file,
    solvers=solvers,
    model=f2f_model,
    nprocs=args.procs,
    transfer_settings=transfer_settings,
    init_transfer=True,
)

# load in the previous design
if args.metal:
    design_out_file = os.path.join(
        base_dir, "design", "ML-metal-sizing.txt" if args.useML else "CF-metal-sizing.txt"
    )
else:
    design_out_file = os.path.join(
        base_dir, "design", "ML-sizing.txt" if args.useML else "CF-sizing.txt"
    )

# reload previous design
f2f_model.read_design_variables_file(comm, design_out_file)


# NOW that we've loaded the loads into a linear static analysis from FUNtoFEM
# let's make a buckling problem with the assembler and load the same loads into that problem
# ------------------------------------------------------------------------------------------

# start like we're making a new static problem
pytacs_builder = pytacs.pyTACS(tacs_aim.root_dat_file, comm, options={})
pytacs_builder.initialize()

# copy the external loads over from the previous assembler
if comm.rank == 0: print(f"here0")
# wait this isn't the right way to add loads into the assembler (goes in the residual?)
pytacs_builder.assembler.applyBCs(solvers.structural.ext_force)
if comm.rank == 0: print(f"rpast here0")

# # except now copy over the assembler in the previous problem (which has the aero loads)
# pytacs_builder.assembler = solvers.structural.assembler

# now create a buckling problem
BP = problems.buckling.BucklingProblem(
    "wing-buckle",
    5.0, # sigma (shift and invert guess eigenvalue)
    10, # num eigenvalues / modes to solve for
    pytacs_builder.assembler,
    comm,
    pytacs_builder.outputViewer,
    pytacs_builder.meshLoader,
    pytacs_builder.isNonlinear,
    {},
)
if comm.rank == 0: print(f"here1")
# Set with original design vars and coordinates, in case they have changed
BP.setDesignVars(pytacs_builder.x0)
if comm.rank == 0: print(f"here2")
BP.setNodes(pytacs_builder.Xpts0)
if comm.rank == 0: print(f"here3")

# now run the BP and write the solution to the caps2tacs / tacsAIM working dir
f2f_ext_force = wing.get_struct_loads(climb, time_index=0)
if comm.rank == 0: print(f"{np.mean(np.abs(f2f_ext_force))=}")
# Fext and aero_ext_force are not the same size for MPI or serial too? Do we need to add the shell DOFs in?
# add shell ext loads - F2F had just (fx,fy,fz) while for shells in TACS we also have (fx,fy,fz,mx,my,mz) the moments too
tacs_shell_ext_force = np.zeros((2*f2f_ext_force.shape[0]))
for i in range(3):
    tacs_shell_ext_force[i::6] = f2f_ext_force[i::3]

BP.solve(Fext=tacs_shell_ext_force)
BP.writeSolution(baseName="full-wing-buckle", outputDir=tacs_aim.root_analysis_dir)