from pathlib import Path
from funtofem.interface import Fun3dModel, Fun3dBC
from mpi4py import MPI
import time

start_time = time.time()

here = Path(__file__).parent
comm = MPI.COMM_WORLD

# Set whether to build an inviscid or viscous mesh
# ------------------------------------------------
# case = "inviscid"  # "turbulent"
# case = "turbulent"
# if case == "inviscid":
#     project_name = "hsct-inviscid"
# else:  # turbulent
#     project_name =
# ------------------------------------------------

# Set up FUN3D model, AIMs, and turn on the flow view
# ------------------------------------------------
fun3d_model = Fun3dModel.build(
    csm_file="hsct.csm",
    comm=comm,
    project_name="hsct-turb",
    volume_mesh="pointwise",
    surface_mesh=None,
)
mesh_aim = fun3d_model.mesh_aim
fun3d_aim = fun3d_model.fun3d_aim
fun3d_aim.set_config_parameter("mode:flow", 1)
fun3d_aim.set_config_parameter("mode:struct", 0)
# ------------------------------------------------

pointwise_aim = mesh_aim.volume_aim
pointwise_aim.main_settings(
    project_name="Wing",
    mesh_format="TECPLOT",  # "AFLR3"
    connector_turn_angle=6.0,
    connector_prox_growth_rate=1.3,
    connector_source_spacing=False,
    domain_algorithm="AdvancingFront",  # "AdvancingFront"
    domain_max_layers=0,
    domain_growth_rate=1.3,
    domain_iso_type="Triangle",
    domain_decay=0.5,
    domain_trex_AR_limit=200,
    domain_wall_spacing=0,
    block_algorithm="AdvancingFront",
    block_boundary_decay=0.5,
    block_collision_buffer=1.0,
    block_max_skew_angle=175,
    block_edge_max_growth_rate=1.8,
    block_full_layers=1,  # if 1 single normals and BL, 0 no BL and multi normals?
    block_max_layers=100,
    block_trex_type="TetPyramidPrismHex",
)

if comm.rank == 0:
    pw_aim = pointwise_aim.aim
    pw_aim.input.Mesh_Sizing = {
        "wingMesh": {"boundaryLayerSpacing": 1e-5},
        "rootUpperEdge": {"numEdgePoints": 100},
        "rootLowerEdge": {"numEdgePoints": 100},
        "tipUpperEdge": {"numEdgePoints": 100},
        "tipLowerEdge": {"numEdgePoints": 100},
    }


FluidMeshOptions = {"pointwiseAIM": {}}

mesh_aim.saveDictOptions(FluidMeshOptions)

Fun3dBC.viscous(caps_group="wing", wall_spacing=1e-5).register_to(fun3d_model)
Fun3dBC.SymmetryY(caps_group="SymmetryY").register_to(fun3d_model)
Fun3dBC.Farfield(caps_group="Farfield").register_to(fun3d_model)
fun3d_model.setup()
fun3d_model.pre_analysis()

dt = time.time() - start_time
print(f"dt seconds = {dt:.4f}", flush=True)
