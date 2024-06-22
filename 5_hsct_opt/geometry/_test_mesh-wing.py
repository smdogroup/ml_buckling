from pathlib import Path
from funtofem.interface import Fun3dModel, Fun3dBC
from mpi4py import MPI

here = Path(__file__).parent
comm = MPI.COMM_WORLD

# Set whether to build an inviscid or viscous mesh
# ------------------------------------------------
# case = "inviscid"  # "turbulent"
case = "turbulent"
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
    surface_mesh="aflr4",
)
mesh_aim = fun3d_model.mesh_aim
fun3d_aim = fun3d_model.fun3d_aim
fun3d_aim.set_config_parameter("mode:flow", 1)
fun3d_aim.set_config_parameter("mode:struct", 0)
# ------------------------------------------------

global_max = 10
global_min = 0.5

mesh_aim.surface_aim.set_surface_mesh(
    ff_growth=1.2,
    mesh_length=1.0,
    min_scale=global_min,
    max_scale=global_max,
    use_aflr4_quads=False,
)

if case == "inviscid":
    mesh_aim.volume_aim.set_boundary_layer(
        initial_spacing=0.005, max_layers=45, thickness=0.01, use_quads=True
    )
    Fun3dBC.inviscid(caps_group="wing").register_to(fun3d_model)
else:
    mesh_aim.volume_aim.set_boundary_layer(
        initial_spacing=0.0001, max_layers=55, thickness=0.01, use_quads=True
    )
    Fun3dBC.viscous(caps_group="wing", wall_spacing=1).register_to(fun3d_model)

FluidMeshOptions = {"aflr4AIM": {}, "aflr3AIM": {}}

FluidMeshOptions["aflr4AIM"]["Mesh_Sizing"] = {
    "leEdgeMesh": {"scaleFactor": 0.08},
    "teEdgeMesh": {"scaleFactor": 0.25},
    "tipEdgeMesh": {"scaleFactor": 0.2},
    "wingMesh": {"scaleFactor": 1.0, "AFLR4_quad_local": 0.0, "min_scale": 0.01},
}

FluidMeshOptions["aflr4AIM"]["curv_factor"] = 0.0001
FluidMeshOptions["aflr4AIM"]["ff_cdfr"] = 1.2
FluidMeshOptions["aflr4AIM"]["mer_all"] = 1

mesh_aim.saveDictOptions(FluidMeshOptions)

if comm.rank == 0:
    aflr3_aim = mesh_aim.volume_aim.aim
    aflr3_aim.input.Mesh_Format = "Tecplot" # so I can view in Tecplot

Fun3dBC.SymmetryY(caps_group="SymmetryY").register_to(fun3d_model)
Fun3dBC.Farfield(caps_group="Farfield").register_to(fun3d_model)
fun3d_model.setup()
fun3d_aim.pre_analysis()
