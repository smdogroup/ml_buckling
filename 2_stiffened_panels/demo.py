"""
Sean Engelstad
: 
Demo analysis of a stiffened panel
"""

import ml_buckling as mlb
from mpi4py import MPI

comm = MPI.COMM_WORLD

geometry = mlb.StiffenedPlateGeometry(
    a=1.0,
    b=1.0,
    h=0.005,
    num_stiff=5,
    w_b=0.02,
    t_b=0.002,
    h_w=0.1,
    t_w=0.003,
)

material = mlb.CompositeMaterial.solvay5320(ply_angle=0)

stiff_analysis = mlb.StiffenedPlateAnalysis(
    comm=comm,
    geometry=geometry,
    stiffener_material=material,
    plate_material=material,
)

stiff_analysis.pre_analysis(
    global_mesh_size=0.1,
    exx=1.0,
    exy=0.0,
    clamped=False,
    edge_pt_min=5,
    edge_pt_max=40,
)
stiff_analysis.run_buckling_analysis(sigma=10.0, num_eig=5, write_soln=True)
stiff_analysis.post_analysis()