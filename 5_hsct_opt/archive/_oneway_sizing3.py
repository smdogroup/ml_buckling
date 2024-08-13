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

comm = MPI.COMM_WORLD

base_dir = os.path.dirname(os.path.abspath(__file__))
csm_path = os.path.join(base_dir, "geometry", "hsct.csm")

# F2F MODEL and SHAPE MODELS
# ----------------------------------------

f2f_model = FUNtoFEMmodel("hsct-sizing4")
tacs_model = caps2tacs.TacsModel.build(
    csm_file=csm_path,
    comm=comm,
    problem_name="capsStruct4",
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

# TODO : code to figure out which OML desvars exist in current internal structure
x1_LE = tacs_aim.get_output_parameter("wing:x1_LE")
x2_LE = tacs_aim.get_output_parameter("wing:x2_LE")
x3_LE = tacs_aim.get_output_parameter("wing:x3_LE")
x4_LE = tacs_aim.get_output_parameter("wing:x4_LE")
x1_TE = tacs_aim.get_output_parameter("wing:x1_TE")
y1 = tacs_aim.get_output_parameter("wing:y1")
y2 = tacs_aim.get_output_parameter("wing:y2")
y3 = tacs_aim.get_output_parameter("wing:y3")
y4 = tacs_aim.get_output_parameter("wing:y4")
sspan = tacs_aim.get_output_parameter("wing:sspan")
z_root = 0.0
z_tip = -sspan

rib_a1 = tacs_aim.get_design_parameter("wing:rib_a1")
rib_a2 = tacs_aim.get_design_parameter("wing:rib_a2")
rib_a3 = 1 - rib_a1 - rib_a2
spar_a1 = tacs_aim.get_design_parameter("wing:spar_a1")
spar_a2 = tacs_aim.get_design_parameter("wing:spar_a2")
spar_a3 = 1 - spar_a1 - spar_a2
ybar1 = tacs_aim.get_design_parameter("wing:ybar1")
ybar2 = tacs_aim.get_design_parameter("wing:ybar2")
ybar3 = 1.0 - ybar1 - ybar2

# some mistake in the math for checking which panels available
# so temporarily just using lists and CSM file geometry
first_panels_dict = {
    1: 1,
    2: 1,
    3: 2,
    4: 2,
    5: 3,
    6: 3,
    7: 4,
    8: 5,
    9: 5,
    10: 6,
    11: 6,
    12: 6,
    13: 6,
    14: 7,
    15: 7,
    16: 7,
    17: 7,
    18: 8,
    19: 8,
}

for iOML in range(1, nOML + 1):
    # name = f"LEtop{iOML}"
    # prop = caps2tacs.ShellProperty(
    #     caps_group=name, material=titanium_alloy, membrane_thickness=0.04
    # ).register_to(tacs_model)
    # Variable.structural(name, value=0.01).set_bounds(
    #     lower=0.001, upper=0.15, scale=100.0
    # ).register_to(wing)

    # name = f"LEbot{iOML}"
    # prop = caps2tacs.ShellProperty(
    #     caps_group=name, material=titanium_alloy, membrane_thickness=0.04
    # ).register_to(tacs_model)
    # Variable.structural(name, value=0.01).set_bounds(
    #     lower=0.001, upper=0.15, scale=100.0
    # ).register_to(wing)

    # FIGURE OUT WHICH OML panels exist in the geometry
    # z/y position of inboard rib of each rib-rib section of an OML panel
    # it's the z position before yz swap and y position after yz swap
    fr1 = (iOML - 1) / (nribs - 1)
    rib_fr = rib_a1 * fr1 + rib_a2 * fr1**2 + rib_a3 * fr1**3
    z_rib = z_root * (1 - rib_fr) + z_tip * rib_fr

    # get the min x position of this trimmed rib (by finding the loft section it belongs to)
    if -y2 <= z_rib <= -y1:
        sec_fr = rib_fr / ybar1
        xmin_rib = x1_LE * (1 - sec_fr) + x2_LE * sec_fr
    elif -y3 <= z_rib <= -y2:
        sec_fr = (rib_fr - ybar1) / (ybar2)
        xmin_rib = x2_LE * (1 - sec_fr) + x3_LE * sec_fr
    else:
        sec_fr = (rib_fr - ybar1 - ybar2) / (ybar3)
        xmin_rib = x3_LE * (1 - sec_fr) + x4_LE * sec_fr

    for ispar in range(1, nspars + 1):
        fr2 = ispar / (nspars + 1)
        spar_fr = fr2 * (spar_a1 + fr2 * (spar_a2 + fr2 * spar_a3))
        x_spar = x1_LE * (1 - spar_fr) + x1_TE * spar_fr
        # if xmin_rib < x_spar:
        # temporarily just using dict to make the check
        if ispar >= first_panels_dict[iOML]:
            # then add these OML groups
            name = f"OMLtop{iOML}-{ispar}"
            prop = caps2tacs.ShellProperty(
                caps_group=name, material=titanium_alloy, membrane_thickness=0.04
            ).register_to(tacs_model)
            Variable.structural(name, value=0.01).set_bounds(
                lower=0.001, upper=0.15, scale=100.0
            ).register_to(wing)

            name = f"OMLbot{iOML}-{ispar}"
            prop = caps2tacs.ShellProperty(
                caps_group=name, material=titanium_alloy, membrane_thickness=0.04
            ).register_to(tacs_model)
            Variable.structural(name, value=0.01).set_bounds(
                lower=0.001, upper=0.15, scale=100.0
            ).register_to(wing)

    # after last spar
    ispar = nspars + 1
    name = f"OMLtop{iOML}-{ispar}"
    prop = caps2tacs.ShellProperty(
        caps_group=name, material=titanium_alloy, membrane_thickness=0.04
    ).register_to(tacs_model)
    Variable.structural(name, value=0.01).set_bounds(
        lower=0.001, upper=0.15, scale=100.0
    ).register_to(wing)

    name = f"OMLbot{iOML}-{ispar}"
    prop = caps2tacs.ShellProperty(
        caps_group=name, material=titanium_alloy, membrane_thickness=0.04
    ).register_to(tacs_model)
    Variable.structural(name, value=0.01).set_bounds(
        lower=0.001, upper=0.15, scale=100.0
    ).register_to(wing)

for prefix in ["LE", "TE"]:
    name = f"{prefix}spar"
    prop = caps2tacs.ShellProperty(
        caps_group=name, material=titanium_alloy, membrane_thickness=0.04
    ).register_to(tacs_model)
    Variable.structural(name, value=0.01).set_bounds(
        lower=0.001, upper=0.15, scale=100.0
    ).register_to(wing)

# add the previous variables too so we can read in from sizing.txt file and get initial designs
# they will be set to inactive later
for iOML in range(1, nOML + 1):
    # no shell properties since these variables are placeholders to process the sizing.txt file
    Variable.structural(f"OML{iOML}", value=0.01).set_bounds(
        lower=0.001, upper=0.15, scale=100.0
    ).register_to(wing)

    # Variable.structural(f"LE{iOML}", value=0.01).set_bounds(
    #     lower=0.001, upper=0.15, scale=100.0
    # ).register_to(wing)

# register the wing body to the model
wing.register_to(f2f_model)

# RELOAD PREVIOUS DESIGN AND MODIFY INITIAL VARIABLES HERE..
# ---------------------------------------------------------
design_in_file = os.path.join(base_dir, "design", "sizing.txt")

# reload previous design
f2f_model.read_design_variables_file(comm, design_in_file)

var_dict = {var.name: var.value for var in f2f_model.get_variables()}

for iOML in range(1, nOML + 1):
    # LE_prev = wing.get_variable(f"LE{iOML}")
    # LEtop = wing.get_variable(f"LEtop{iOML}")
    # LEbot = wing.get_variable(f"LEbot{iOML}")
    # LEtop.value = LE_prev.value
    # LEbot.value = LE_prev.value
    # LE_prev.active = False

    OML_prev = wing.get_variable(f"OML{iOML}")
    for ispar in range(1, nspars + 2):
        OMLtop = wing.get_variable(f"OMLtop{iOML}-{ispar}")
        OMLbot = wing.get_variable(f"OMLbot{iOML}-{ispar}")
        if (
            OMLtop is None or OMLbot is None
        ):  # not all panels exist (might be trimmed off)
            continue
        OMLtop.value = OML_prev.value
        OMLbot.value = OML_prev.value
    OML_prev.active = False

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
climb.include(mass).include(ksfailure)
climb.set_temperature(T_ref=216, T_inf=216)
climb.set_flow_ref_vals(qinf=3.16e4)
climb.register_to(f2f_model)

# COMPOSITE FUNCTIONS
# -------------------------------------------------------

# skin thickness adjacency constraints
variables = f2f_model.get_variables()
for irib in range(1, nribs):  # [1, nribs-1] since right exclusive
    left_var = f2f_model.get_variables(names=f"rib{irib}")
    right_var = f2f_model.get_variables(names=f"rib{irib+1}")
    adj_constr = (left_var - right_var) / left_var
    adj_constr.set_name(f"rib{irib}-{irib+1}").optimize(
        lower=-0.15, upper=0.15, scale=1.0, objective=False
    ).register_to(f2f_model)

for iOML in range(1, nOML):  # [1, nOML-1] since right exclusive
    for side in ["top", "bot"]:
        # Leading edge panel constraints
        # left_var = f2f_model.get_variables(names=f"LE{side}{iOML}")
        # right_var = f2f_model.get_variables(names=f"LE{side}{iOML+1}")
        # adj_constr = (left_var - right_var) / left_var
        # adj_ratio = 0.15
        # adj_constr.set_name(f"LE{side}{iOML}").optimize(
        #     lower=-adj_ratio, upper=adj_ratio, scale=1.0, objective=False
        # ).register_to(f2f_model)

        # OML panel spanwise adjacency constraints
        for ispar in range(1, nspars + 2):  # [1, nspars+1] since right exclusive
            left_var = f2f_model.get_variables(names=f"OML{side}{iOML}-{ispar}")
            right_var = f2f_model.get_variables(names=f"OML{side}{iOML+1}-{ispar}")
            if (
                left_var is None or right_var is None
            ):  # some variables were trimmed away
                continue
            adj_constr = left_var - right_var
            # CHANGE ADJACENCY constraints to linear so the optimizer always satisfies them (less oscillation)
            adj_constr = (left_var - right_var) / left_var
            # assume gauge about 2mm here
            adj_constr.set_name(f"OML{side}{iOML}_{iOML+1}-{ispar}").optimize(
                lower=-0.15, upper=0.15, scale=1.0, objective=False
            ).register_to(f2f_model)

for iOML in range(1, nOML + 1):  # [1, nOML] since right exclusive
    for side in ["top", "bot"]:
        # OML panel chordwise adjacency constraints
        for ispar in range(1, nspars + 1):  # [1, nspars] since right exclusive
            left_var = f2f_model.get_variables(names=f"OML{side}{iOML}-{ispar}")
            right_var = f2f_model.get_variables(names=f"OML{side}{iOML}-{ispar+1}")
            if (
                left_var is None or right_var is None
            ):  # some variables were trimmed away
                continue
            # CHANGE ADJACENCY constraints to linear so the optimizer always satisfies them (less oscillation_
            adj_constr = (left_var - right_var) / left_var
            adj_constr.set_name(f"OML{side}{iOML}-{ispar}_{ispar+1}").optimize(
                lower=-0.15, upper=0.15, scale=1.0, objective=False
            ).register_to(f2f_model)


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
    nprocs=8,
    transfer_settings=transfer_settings,
)

# PYOPTSPARSE OPTMIZATION
# -------------------------------------------------------------

# create an OptimizationManager object for the pyoptsparse optimization problem
design_out_file = os.path.join(base_dir, "design", "all_sizing.txt")

manager = OptimizationManager(
    tacs_driver, design_out_file=design_out_file, hot_start=hot_start, debug=True
)

# create the pyoptsparse optimization problem
opt_problem = Optimization("hsctOpt", manager.eval_functions)

# add funtofem model variables to pyoptsparse
manager.register_to_problem(opt_problem)

# run an SNOPT optimization
snoptimizer = SLSQP(options={"IPRINT": 1})

design_folder = os.path.join(base_dir, "design")
if not os.path.exists(design_folder):
    os.mkdir(design_folder)
history_file = os.path.join(design_folder, "all_sizing.hst")
store_history_file = history_file if store_history else None
hot_start_file = history_file if hot_start else None


sol = snoptimizer(
    opt_problem,
    sens=manager.eval_gradients,
    storeHistory=store_history_file,
    hotStart=hot_start_file,
)

# print final solution
sol_xdict = sol.xStar
print(f"Final solution = {sol_xdict}", flush=True)
