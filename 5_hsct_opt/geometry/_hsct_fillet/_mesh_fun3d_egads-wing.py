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
egads_aim.input.Edge_Point_Max = 40
egads_aim.input.Mesh_Elements = "Mixed"
egads_aim.input.Tess_Params = [0.5, 0.1, 15]

# was freezing very badly while making the surface mesh
# so trying to set individual mesh sizing parameters now
num_pts_up = 200
num_pts_bot = num_pts_up

refine_tanh = True

if refine_tanh:
    egads_aim.input.Mesh_Sizing = {
        "Farfield": {"tessParams": [5.0, 0.1, 20.0]},
        "SymmetryY": {"tessParams": [4.0, 0.1, 20.0]},
        "wingMesh": {"tessParams": [0.01, 0.01, 1.0],  # [0.5, 0.1, 15.0]
                    #  "Edge_Point_Min" : 20, "Edge_Point_Max" : 50
        },
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
    }
else:
    egads_aim.input.Mesh_Sizing = {
        "Farfield": {"tessParams": [5.0, 0.1, 20.0]},
        "SymmetryY": {"tessParams": [4.0, 0.1, 20.0]},
        "wingMesh": {"tessParams": [0.1, 0.01, 4.0],  # [0.5, 0.1, 15.0]
                    #  "Edge_Point_Min" : 20, "Edge_Point_Max" : 50
        },
        "tipUpperEdge": {
            "numEdgePoints": num_pts_up,
        },
        "tipLowerEdge": {
            "numEdgePoints": num_pts_bot,
        },
        "rootUpperEdge": {
            "numEdgePoints": num_pts_up,
        },
        "rootLowerEdge": {
            "numEdgePoints": num_pts_bot,
        },
    }

aflr3_aim.input.Mesh_Format = "TECPLOT"
aflr3_aim.input.BL_Initial_Spacing = 1e-5
aflr3_aim.input.BL_Thickness = 1e-3
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
    "wing" : {"bcType" : "viscous", "boundaryLayerSpacing" : 1e-5},
    "Farfield" : {"bcType" : "Farfield"},
    "SymmetryY" : "SymmetryY",
}

fun3d_aim.preAnalysis()

dt = time.time() - start_time
print(f"dt seconds = {dt:.4f}", flush=True)
