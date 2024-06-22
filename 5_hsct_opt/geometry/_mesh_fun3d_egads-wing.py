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
case = "inviscid"
if case == "inviscid":
    project_name = "hsct-inviscid"
else:  # turbulent
    project_name = "hsct-turb"
# ------------------------------------------------

# Set up FUN3D model, AIMs, and turn on the flow view
# ------------------------------------------------
fun3d_model = Fun3dModel.build(
    csm_file="hsct.csm",
    comm=comm,
    project_name=project_name,
    volume_mesh="aflr3",
    surface_mesh="egads",
)
mesh_aim = fun3d_model.mesh_aim
fun3d_aim = fun3d_model.fun3d_aim
fun3d_aim.set_config_parameter("mode:flow", 1)
fun3d_aim.set_config_parameter("mode:struct", 0)
# ------------------------------------------------

mesh_aim.surface_aim.set_surface_mesh(
    edge_pt_min=10,
    edge_pt_max=40,
    mesh_elements="Mixed",
    global_mesh_size=0.5,
    max_surf_offset=0.1,
    max_dihedral_angle=15,
)

# was freezing very badly while making the surface mesh
# so trying to set individual mesh sizing parameters now
num_pts_up = 80
num_pts_bot = num_pts_bot = 80
if comm.rank == 0:
    egads_tess_aim = mesh_aim.surface_aim.aim
    egads_tess_aim.input.Mesh_Sizing = {
        "Farfield": {"tessParams": [5.0, 0.1, 20.0]},
        "SymmetryY": {"tessParams": [4.0, 0.1, 20.0]},
        "wingMesh": {"tessParams": [0.5, 0.1, 15.0], 
                    #  "Edge_Point_Min" : 20, "Edge_Point_Max" : 50
        },
        # "rootEdgeMesh": {
        # "numEdgePoints": 80,
        # },
        # "tipEdgeMesh": {
        #     "numEdgePoints": 80,
        # },
        # "tipUpperEdge": {
        #     "numEdgePoints": num_pts_up,
        #     "edgeDistribution": "Tanh",
        #     "initialNodeSpacing": [0.005, 0.002],
        # },
        # "tipLowerEdge": {
        #     "numEdgePoints": num_pts_bot,
        #     "edgeDistribution": "Tanh",
        #     "initialNodeSpacing": [0.002, 0.005],
        # },
        # "rootUpperEdge": {
        #     "numEdgePoints": num_pts_up,
        #     "edgeDistribution": "Tanh",
        #     "initialNodeSpacing": [0.005, 0.002],
        # },
        # "rootLowerEdge": {
        #     "numEdgePoints": num_pts_bot,
        #     "edgeDistribution": "Tanh",
        #     "initialNodeSpacing": [0.002, 0.005],
        # },
        # "rootEdgeMesh": {
        # "numEdgePoints": 80,
        #     "edgeDistribution": "Tanh",
        #     "initialNodeSpacing": [0, 0.05],
        # },
        # "tipEdgeMesh": {
        #     "numEdgePoints": 80,
        #     "edgeDistribution": "Tanh",
        #     "initialNodeSpacing": [0, 0.05],
        # },
    }

if case == "inviscid":
    mesh_aim.volume_aim.set_boundary_layer(
        initial_spacing=0.005, max_layers=45, thickness=0.01, use_quads=True
    )
    Fun3dBC.inviscid(caps_group="wing").register_to(fun3d_model)
else:
    mesh_aim.volume_aim.set_boundary_layer(
        initial_spacing=1e-5, max_layers=100, thickness=0.01, use_quads=True
    )
    Fun3dBC.viscous(caps_group="wing", wall_spacing=1e-5).register_to(fun3d_model)

FluidMeshOptions = {"egadsTessAIM": {}, "aflr3AIM": {}}

#FluidMeshOptions["egadsTessAIM"]["Mesh_Sizing"] = {
#    "leEdgeMesh": {"scaleFactor": 0.08},
#    "teEdgeMesh": {"scaleFactor": 0.25},
#    "tipEdgeMesh": {"scaleFactor": 0.2},
#    "wingMesh": {"scaleFactor": 1.0},
#}

mesh_aim.saveDictOptions(FluidMeshOptions)

Fun3dBC.SymmetryY(caps_group="SymmetryY").register_to(fun3d_model)
Fun3dBC.Farfield(caps_group="Farfield").register_to(fun3d_model)
fun3d_model.setup()
fun3d_aim.pre_analysis()

dt = time.time() - start_time
print(f"dt seconds = {dt:.4f}", flush=True)
