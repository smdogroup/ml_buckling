"""
Sean P. Engelstad, Georgia Tech 2024

Local machine optimization for the panel thicknesses using all OML and LE panels (more design variables) but no shape variables
use `mpirun -n 3 python build_exploded_mesh.py` to run this script (3 procs needed for each exploded mesh)
"""

from funtofem import *

# import openmdao.api as om
from mpi4py import MPI
from tacs import caps2tacs
import os

comm = MPI.COMM_WORLD

base_dir = os.path.dirname(os.path.abspath(__file__))
csm_path = os.path.join(base_dir, "..", "..", "geometry", "hsct.csm")

import sys

sys.path.append("../../")
from _closed_form_callback import closed_form_callback as callback

nprocs = 3

# F2F MODEL and SHAPE MODELS
# ----------------------------------------

f2f_model = FUNtoFEMmodel("temp")
tacs_model = caps2tacs.TacsModel.build(
    csm_file=csm_path,
    comm=comm,
    problem_name="capsExploded",
    active_procs=[_ for _ in range(nprocs)],
    verbosity=1,
)
tacs_model.mesh_aim.set_mesh(  # need a refined-enough mesh for the derivative test to pass
    edge_pt_min=2,
    edge_pt_max=300,
    mesh_elements="Mixed",
    global_mesh_size=0.006,  # 0.004 was 130k mesh
    max_surf_offset=0.01,
    max_dihedral_angle=5,
).register_to(
    tacs_model
)
f2f_model.structural = tacs_model

# run this with 3 procs in mpi and this should work!
# 0 (off), 1 (upperOML), 2 (internalStruc), 3(lowerOML)
exploded_view = comm.rank + 1

tacs_aim = tacs_model.tacs_aim
tacs_aim.set_config_parameter("mode:flow", 0)
tacs_aim.set_config_parameter("mode:struct", 1)
tacs_aim.set_config_parameter("wing:allOMLgroups", 1)
tacs_aim.set_config_parameter("wing:includeLE", 0)
tacs_aim.set_config_parameter("wing:includeTE", 0)
tacs_aim.set_config_parameter("wing:nspars", 40)  # same # as concorde
tacs_aim.geometry.cfgpmtr["wing:exploded"].value = exploded_view


for proc in tacs_aim.active_procs:
    if comm.rank == proc:
        egads_aim = tacs_model.mesh_aim.aim
        # turn off quading for OML faces
        egads_aim.input.TFI_Templates = False
        # turn off all other TFI_Templates for internal struct
        # and set # elems to 8 in Z direction [vert is half/midplane edges]
        # egads_aim.input.Mesh_Sizing = {
        #     "rib" : {"TFI_Templates" : True},
        #     "spar" : {"TFI_Templates" : True},
        #     "vert": {"numEdgePoints": 4},
        #     "vert2" : {"numEdgePoints" : 6},
        # }

# add tacs constraints in
caps2tacs.PinConstraint("dummy").register_to(tacs_model)
# caps2tacs.PinConstraint("root").register_to(tacs_model)
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

# NOTE : the feature in ESP/CAPS that gives you the names of all capsGroups
# failed to list all of them since it hit the character limit in the textbox
# so I basically had to count the ranges of all capsGroups myself..
component_groups = []

if exploded_view == 1:
    prefix_list = ["OMLtop"]
elif exploded_view == 3:
    prefix_list = ["OMLbot"]
else:
    prefix_list = ["OMLtop"]

for prefix in prefix_list:
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

# add rib component groups based on OML ones
if exploded_view == 2:
    rib_groups = ["rib" + comp[6:] for comp in component_groups if "OMLtop" in comp]
    rib_groups += [f"rib20-{ispar}" for ispar in range(35, 41 + 1)]
    component_groups = rib_groups  # overwrite not += so only does ribs

# add spar component groups
if exploded_view == 2:
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

for icomp, comp in enumerate(component_groups):
    caps2tacs.CompositeProperty.null(comp, null_material).register_to(tacs_model)

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

f2f_model.read_design_variables_file(comm, "CF-sizing.txt")

# exit()

# INITIAL STRUCTURE MESH, SINCE NO STRUCT SHAPE VARS
# --------------------------------------------------

tacs_aim.setup_aim()
tacs_aim.pre_analysis()

# # DISCIPLINE INTERFACES AND DRIVERS
# # -----------------------------------------------------

# from funtofem import TacsPanelDimensions, TacsOutputGenerator, TacsSteadyInterface, TransferSettings, OnewayStructDriver
# from tacs import pytacs
# import numpy as np

# for proc in tacs_aim.active_procs:
#     if comm.rank == proc:
#         # run the TACS analysis in each capsExploded_{comm.rank} directory

#         solvers = SolverManager(comm)

#         # manually build the TacsSteadyInterface
#         # Split the communicator
#         world_rank = comm.rank
#         if comm.rank == proc:
#             color = 1
#         else:
#             color = MPI.UNDEFINED
#         tacs_comm = comm.Split(color, world_rank)

#         assembler = None
#         f5 = None
#         Fvec = None
#         tacs_panel_dimensions = TacsPanelDimensions(
#             comm=comm,
#             panel_length_dv_index=0,
#             panel_width_dv_index=5,
#         )
#         if world_rank < nprocs:
#             # Create the assembler class
#             fea_assembler = pytacs.pyTACS(tacs_aim.root_dat_file, tacs_comm, options={})

#             """
#             Automatically adds structural variables from the BDF / DAT file into TACS
#             as long as you have added them with the same name in the DAT file.

#             Uses a custom funtofem callback to create thermoelastic shells which are unavailable
#             in pytacs default callback. And creates the DVs in the correct order in TACS based on DVPREL cards.
#             """

#             # get dict of struct DVs from the bodies and structural variables
#             # only supports thickness DVs for the structure currently
#             structDV_dict = {}
#             variables = f2f_model.get_variables()
#             structDV_names = []

#             # Get the structural variables from the global list of variables.
#             struct_variables = []
#             for var in variables:
#                 if var.analysis_type == "structural":
#                     struct_variables.append(var)
#                     structDV_dict[var.name.lower()] = var.value
#                     structDV_names.append(var.name.lower())

#             # Set up constitutive objects and elements in pyTACS
#             fea_assembler.initialize(callback)

#             # if panel_length_dv_index is not None:
#             tacs_panel_dimensions.panel_length_constr = (
#                 fea_assembler.createPanelLengthConstraint("PanelLengthCon")
#             )
#             tacs_panel_dimensions.panel_length_constr.addConstraint(
#                 TacsSteadyInterface.LENGTH_CONSTR,
#                 dvIndex=tacs_panel_dimensions.panel_length_dv_index,
#             )

#             # if panel_width_dv_index is not None:
#             tacs_panel_dimensions.panel_width_constr = (
#                 fea_assembler.createPanelWidthConstraint("PanelWidthCon")
#             )
#             tacs_panel_dimensions.panel_width_constr.addConstraint(
#                 TacsSteadyInterface.WIDTH_CONSTR, dvIndex=tacs_panel_dimensions.panel_width_dv_index
#             )

#             # Retrieve the assembler from pyTACS fea_assembler object
#             assembler = fea_assembler.assembler

#             # Set the output file creator
#             f5 = fea_assembler.outputViewer

#         # Create the output generator
#         gen_output = TacsOutputGenerator(prefix, f5=f5)

#         # get struct ids for coordinate derivatives and .sens file
#         struct_id = None
#         if assembler is not None:
#             # get list of local node IDs with global size, with -1 for nodes not owned by this proc
#             num_nodes = fea_assembler.meshLoader.bdfInfo.nnodes
#             bdfNodes = range(num_nodes)
#             local_tacs_ids = fea_assembler.meshLoader.getLocalNodeIDsFromGlobal(
#                 bdfNodes, nastranOrdering=False
#             )

#             """
#             the local_tacs_ids list maps nastran nodes to tacs indices with:
#                 local_tacs_ids[nastran_node-1] = local_tacs_id
#             Only a subset of all tacs ids are owned by each processor
#                 note: tacs_ids in [0, #local_tacs_ids], #local_tacs_ids <= nnodes
#                 for nastran nodes not on this processor, local_tacs_id[nastran_node-1] = -1

#             The next lines of code invert this map to the list 'struct_id' with:
#                 struct_id[local_tacs_id] = nastran_node

#             This is then later used by funtofem_model.write_sensitivity_file method to write
#             ESP/CAPS nastran_CAPS.sens files for the tacsAIM to compute shape derivatives
#             """

#             # get number of non -1 tacs ids, total number of actual tacs_ids
#             n_tacs_ids = len([tacs_id for tacs_id in local_tacs_ids if tacs_id != -1])

#             # reverse the tacs id to nastran ids map since we want tacs_id => nastran_id - 1
#             nastran_ids = np.zeros((n_tacs_ids), dtype=np.int64)
#             for nastran_id_m1, tacs_id in enumerate(local_tacs_ids):
#                 if tacs_id != -1:
#                     nastran_ids[tacs_id] = int(nastran_id_m1 + 1)

#             # convert back to list of nastran_ids owned by this processor in order
#             struct_id = list(nastran_ids)

#         # We might need to clean up this code. This is making educated guesses
#         # about what index the temperature is stored. This could be wrong if things
#         # change later. May query from TACS directly?
#         thermal_idx = -1
#         if assembler is not None and thermal_index == -1:
#             varsPerNode = assembler.getVarsPerNode()

#             # This is the likely index of the temperature variable
#             if varsPerNode == 1:  # Thermal only
#                 thermal_index = 0
#             elif varsPerNode == 4:  # Solid + thermal
#                 thermal_index = 3
#             elif varsPerNode >= 7:  # Shell or beam + thermal
#                 thermal_index = varsPerNode - 1

#         # Broad cast the thermal index to ensure it's the same on all procs
#         thermal_index = comm.bcast(thermal_index, root=0)

#         # Create the tacs interface
#         solvers.structural = TacsSteadyInterface(
#             comm,
#             f2f_model,
#             assembler,
#             gen_output,
#             struct_id=struct_id,
#             tacs_comm=tacs_comm,
#             tacs_panel_dimensions=tacs_panel_dimensions,
#         )

#         # read in aero loads
#         aero_loads_file = os.path.join(os.getcwd(), "cfd", "loads", "uncoupled_turb_loads.txt")

#         transfer_settings = TransferSettings(npts=200)

#         # build the shape driver from the file
#         tacs_driver = OnewayStructDriver.prime_loads_from_file(
#             filename=aero_loads_file,
#             solvers=solvers,
#             model=f2f_model,
#             nprocs=1,
#             transfer_settings=transfer_settings,
#             init_transfer=True,
#         )

#         # f2f_model.read_design_variables_file(comm, "design/CF-sizing.txt")

#         tacs_driver.solve_forward()
