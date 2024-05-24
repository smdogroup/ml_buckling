from pathlib import Path
from funtofem.interface import Fun3dModel, Fun3dBC
from mpi4py import MPI

here = Path(__file__).parent
comm = MPI.COMM_WORLD

# Set whether to build an inviscid or viscous mesh
# ------------------------------------------------
# case = "inviscid"
case = "turbulent"

project_name = "gbm-half"
# ------------------------------------------------

# Set up FUN3D model, AIMs, and turn on the flow view
# ------------------------------------------------
fun3d_model = Fun3dModel.build(
    csm_file="gbm-half.csm",
    comm=comm,
    project_name=project_name,
    problem_name="capsFluidEgads-half",
    volume_mesh="aflr3",
    surface_mesh="egads",
    verbosity=0,
)
mesh_aim = fun3d_model.mesh_aim
fun3d_aim = fun3d_model.fun3d_aim
fun3d_aim.set_config_parameter("view:flow", 1)
fun3d_aim.set_config_parameter("view:struct", 0)

# if comm.rank == 0:
#    aflr3_aim = mesh_aim.volume_aim.aim
#    aflr3_aim.input.Mesh_Format = "AFLR3"

# ------------------------------------------------

mesh_aim.surface_aim.set_surface_mesh(
    edge_pt_min=3,
    edge_pt_max=200,
    mesh_elements="Mixed",
    global_mesh_size=0,
    max_surf_offset=0.0008,
    max_dihedral_angle=15,
)

num_pts_up = 60
num_pts_bot = 60
num_pts_y = 70
num_finite_te = 4

horiz_spacing = [0.08, 0.05]

mesh_aim.surface_aim.aim.input.Mesh_Sizing = {
    "teHorizEdgeMesh": {
        "numEdgePoints": num_pts_y,
        "edgeDistribution": "Tanh",
        "initialNodeSpacing": horiz_spacing,
    },
    "teVertEdgeMesh": {
        "numEdgePoints": num_finite_te,
    },
    "farfieldEdgeMesh": {
        "numEdgePoints": 8,
    },
    "leEdgeMesh": {
        "numEdgePoints": num_pts_y,
        "edgeDistribution": "Tanh",
        "initialNodeSpacing": horiz_spacing,
    },
    "tipUpperEdgeMesh": {
        "numEdgePoints": num_pts_up,
        "edgeDistribution": "Tanh",
        "initialNodeSpacing": [0.005, 0.002],
    },
    "tipLowerEdgeMesh": {
        "numEdgePoints": num_pts_bot,
        "edgeDistribution": "Tanh",
        "initialNodeSpacing": [0.002, 0.005],
    },
    "rootUpperEdgeMesh": {
        "numEdgePoints": num_pts_up,
        "edgeDistribution": "Tanh",
        "initialNodeSpacing": [0.005, 0.002],
    },
    "rootLowerEdgeMesh": {
        "numEdgePoints": num_pts_bot,
        "edgeDistribution": "Tanh",
        "initialNodeSpacing": [0.002, 0.005],
    },
    "tipMesh": {"tessParams": [0.05, 0.01, 20.0]},
}

Fun3dBC.Farfield(caps_group="farfield").register_to(fun3d_model)
Fun3dBC.SymmetryY(caps_group="symmetry").register_to(fun3d_model)

if case == "inviscid":
    Fun3dBC.inviscid(caps_group="wing").register_to(fun3d_model)
else:
    mesh_aim.volume_aim.set_boundary_layer(
        initial_spacing=1e-5, max_layers=60, thickness=0.25, use_quads=True
    )
    Fun3dBC.viscous(caps_group="wing", wall_spacing=0.1).register_to(fun3d_model)

fun3d_model.setup()
fun3d_aim.pre_analysis()
