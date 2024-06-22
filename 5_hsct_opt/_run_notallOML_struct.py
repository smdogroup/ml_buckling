"""
Sean P. Engelstad, Georgia Tech 2023

Local machine optimization for the panel thicknesses using nribs-1 OML panels and nribs-1 LE panels
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
from _closed_form_callback import closed_form_callback

comm = MPI.COMM_WORLD

base_dir = os.path.dirname(os.path.abspath(__file__))
csm_path = os.path.join(base_dir, "geometry", "hsct.csm")

# F2F MODEL and SHAPE MODELS
# ----------------------------------------

f2f_model = FUNtoFEMmodel("hsct-sizing")
tacs_model = caps2tacs.TacsModel.build(
    csm_file=csm_path,
    comm=comm,
    problem_name="capsStruct2",
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
        }
        # "LEribFace": {"tessParams": [0.03, 0.1, 3]},
        #    "LEribEdge": {"numEdgePoints" : 20},


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

component_groups = [f"rib{irib}" for irib in range(1, nribs+1)]
component_groups += [f"spar{ispar}" for ispar in range(1, nspars+1)]
component_groups += [f"OML{iOML}" for iOML in range(1, nOML+1)]
component_groups += [f"{prefix}spar" for prefix in ["LE", "TE"]]

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
mass = Function.mass().optimize(
    scale=1.0e-4, objective=True, plot=True, plot_name="mass"
)
ksfailure = Function.ksfailure(ks_weight=10.0, safety_factor=1.5).optimize(
    scale=1.0, upper=1.0, objective=False, plot=True, plot_name="ks-climb"
)
climb.include(ksfailure).include(mass)
climb.set_temperature(T_ref=216, T_inf=216)
climb.set_flow_ref_vals(qinf=3.16e4)
climb.register_to(f2f_model)

# COMPOSITE FUNCTIONS
# ------------------------------------------------------
# NONE for just running struct analysis

# DISCIPLINE INTERFACES AND DRIVERS
# -----------------------------------------------------

solvers = SolverManager(comm)
# solvers.flow = Fun3dInterface(comm, f2f_model, fun3d_dir="meshes")
solvers.structural = TacsSteadyInterface.create_from_bdf(
    model=f2f_model,
    comm=comm,
    nprocs=8,
    callback=closed_form_callback,
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
    nprocs=8,
    transfer_settings=transfer_settings,
)

tacs_driver.solve_forward()
