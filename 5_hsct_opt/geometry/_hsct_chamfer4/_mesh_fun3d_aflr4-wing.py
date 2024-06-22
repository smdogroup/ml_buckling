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
aflr4_aim = caps_problem.analysis.create(aim="aflr4AIM", name="aflr4")

for aim in [fun3d_aim, aflr3_aim, aflr4_aim]:
    aim.input.Proj_Name = "hsct-turb"

fun3d_aim.geometry.cfgpmtr["mode:flow"].value = 1
fun3d_aim.geometry.cfgpmtr["mode:struct"].value = 0
fun3d_aim.input.Overwrite_NML = False
# ------------------------------------------------

aflr4_aim.input.ff_cdfr  = 1.10 #1.05
aflr4_aim.input.min_scale = 0.01
aflr4_aim.input.max_scale = 1000.0
aflr4_aim.input.Mesh_Length_Factor = 1 # reference length for unit conversion (not needed)
aflr4_aim.input.mer_all = 1
aflr4_aim.input.Mesh_Gen_Input_String = "mquad=1 mpp=3"
aflr4_aim.input.EGADS_Quad = True #True, was really slow at building EGADS Quadding

wing_c1 = 48.0
ff_full = 100 * wing_c1
ff_size = ff_full / 10.0

edge_SF = 0.05

aflr4_aim.input.Mesh_Sizing = {
    "Farfield": {"scaleFactor" : ff_size},
    "OuterSymmetry": {"scaleFactor" : ff_size},
    "InnerSymmetry": {"scaleFactor" : ff_size/5.0},
    #"wingMesh": {"scaleFactor" : 0.5 }, #0.1
    # "wingTip" : {"scaleFactor" : 0.5}, # default wingScale is 1.0 and is probably too coarse..

    # "tipUpperEdge": { "AFLR4_Edge_Refinement_Weight" : edge_SF},
    # "tipLowerEdge": { "AFLR4_Edge_Refinement_Weight" : edge_SF},
    # "rootUpperEdge": { "AFLR4_Edge_Refinement_Weight" : edge_SF},
    # "rootLowerEdge": { "AFLR4_Edge_Refinement_Weight" : edge_SF},
    # "leadingEdge": { "AFLR4_Edge_Refinement_Weight" : edge_SF},
    # "trailingUpperEdge": { "AFLR4_Edge_Refinement_Weight" : edge_SF},
    # "trailingLowerEdge": { "AFLR4_Edge_Refinement_Weight" : edge_SF},
}

aflr3_aim.input.Mesh_Format = "TECPLOT"
aflr3_aim.input.BL_Initial_Spacing = 1e-6
aflr3_aim.input.BL_Thickness = 1e-4
#aflr3_aim.input.BL_Max_Layers = 
aflr3_aim.input.Mesh_Gen_Input_String = "-blc3"

#fun3d_aim.input.Mesh_Format = "TECPLOT"

aflr3_aim.input["Surface_Mesh"].link(
    aflr4_aim.output["Surface_Mesh"]
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
