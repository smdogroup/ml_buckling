from pathlib import Path
import pyCAPS
from mpi4py import MPI
import time

start_time = time.time()

here = Path(__file__).parent
comm = MPI.COMM_WORLD
# assume serial
# ------------------------------------------------

# Set up FUN3D model, AIMs, and turn on the flow view
# ------------------------------------------------
caps_problem = pyCAPS.Problem(
    problemName="capsFluid", capsFile="hsct.csm", outLevel=1
)
fun3d_aim = caps_problem.analysis.create(aim="fun3dAIM", name="fun3d")
aflr3_aim = caps_problem.analysis.create(aim="aflr3AIM", name="aflr3")
egads_aim = caps_problem.analysis.create(aim="egadsTessAIM", name="egads")

for aim in [fun3d_aim, aflr3_aim, egads_aim]:
    aim.input.Proj_Name = "hsct-turb"

fun3d_aim.geometry.cfgpmtr["mode:flow"].value = 1
fun3d_aim.geometry.cfgpmtr["mode:struct"].value = 0
fun3d_aim.input.Overwrite_NML = False
# ------------------------------------------------
egads_aim.input.Edge_Point_Min = 10
egads_aim.input.Edge_Point_Max = 100
egads_aim.input.Mesh_Elements = "Tri" # "Quad", "Mixed"
egads_aim.input.Tess_Params = [0.1, 0.1, 15]

# was freezing very badly while making the surface mesh
# so trying to set individual mesh sizing parameters now
num_pts_up = 100
num_pts_bot = num_pts_up
num_pts_span = 150

wing_c1 = 48.0
ff_full = 100 * wing_c1
ff_size = ff_full / 10.0

egads_aim.input.Mesh_Sizing = {
    "Farfield": {"tessParams": [ff_size, 0.1, 20.0]},
    "SymmetryY": {"tessParams": [ff_size, 0.1, 20.0]},
    # "InnerSymmetry" : {"tessParams" : [3 * wing_c1, 0.1, 15.0]},
    # "cylinderEdge" : {"numEdgePoints" : int(num_pts_up/2.0)},
    "wingMesh": {"tessParams": [0.05, 0.01, 4.0],  # [0.5, 0.1, 15.0]
                #  "Edge_Point_Min" : 20, "Edge_Point_Max" : 50
    },
    "wingTip" : {"tessParams": [0.03, 0.01, 4.0]},
    "tipUpperEdge": {
        "numEdgePoints": num_pts_up,
        "edgeDistribution": "Tanh",
        "initialNodeSpacing": [0.005, 0.002],
    },
    "tipLowerEdge": {
        "numEdgePoints": num_pts_bot,
        "edgeDistribution": "Tanh",
        "initialNodeSpacing": [0.002, 0.005],
    },
    "rootUpperEdge": {
        "numEdgePoints": num_pts_up,
        "edgeDistribution": "Tanh",
        "initialNodeSpacing": [0.005, 0.002],
    },
    "rootLowerEdge": {
        "numEdgePoints": num_pts_bot,
        "edgeDistribution": "Tanh",
        "initialNodeSpacing": [0.002, 0.005],
    },
    "leadingEdge": {
        "numEdgePoints": num_pts_span,
    },
    "trailingUpperEdge": {
        "numEdgePoints": num_pts_span,
    },
    "trailingLowerEdge": {
        "numEdgePoints": num_pts_span,
    },
}

# egads_aim.runAnalysis()
# egads_aim.geometry.view()
# exit()

aflr3_aim.input.Mesh_Format = "TECPLOT"
aflr3_aim.input.BL_Initial_Spacing = 1e-6
aflr3_aim.input.BL_Thickness = 1e-4
#aflr3_aim.input.BL_Max_Layers = 
aflr3_aim.input.Mesh_Gen_Input_String = "-blc3"

#fun3d_aim.input.Mesh_Format = "TECPLOT"

aflr3_aim.input["Surface_Mesh"].link(
    egads_aim.output["Surface_Mesh"]
)

fun3d_aim.input["Mesh"].link(
    aflr3_aim.output["Volume_Mesh"]
)

fun3d_aim.input.Boundary_Condition = {
    "wing" : {"bcType" : "viscous", "boundaryLayerSpacing" : 1e-6},
    "Farfield" : {"bcType" : "Farfield"},
    "SymmetryY" : "SymmetryY",
}

fun3d_aim.preAnalysis()

dt = time.time() - start_time
print(f"dt seconds = {dt:.4f}", flush=True)
