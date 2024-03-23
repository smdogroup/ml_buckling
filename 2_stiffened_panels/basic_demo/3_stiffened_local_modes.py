"""
Sean Engelstad
: 
Demo analysis of a stiffened panel
"""

import ml_buckling as mlb
from mpi4py import MPI

comm = MPI.COMM_WORLD

# I think what happens when h becomes very low for large a,b the norm of K is too small and that affects the soln
# to prevent low thickness problems => just make overall plate size smaller

"""
Need to make the stiffener stronger w.r.t. the plate
To exaggarate: have really thick stiffener (non-physically thick)
"""

# maybe I'm not correctly modeling the base of the blade stiffener
# I just increase the thickness in this region.. (but maybe it should act like an extra ply or mated there)
# maybe I need a lower aspect ratio plate to get local modes ??

geometry = mlb.StiffenedPlateGeometry(
    a=0.1, 
    b=0.3,
    h=3e-3,
    num_stiff=10,
    w_b=6e-3,
    t_b=3e-3,
    h_w=5e-3,
    t_w=1e-2, # if the wall thickness is too low => stiffener crimping failure happens
)

material = mlb.CompositeMaterial.solvay5320(ply_angle=0)

stiff_analysis = mlb.StiffenedPlateAnalysis(
    comm=comm,
    geometry=geometry,
    stiffener_material=material,
    plate_material=material,
)

stiff_analysis.pre_analysis(
    global_mesh_size=0.03,
    exx=stiff_analysis.affine_exx,
    exy=0.0,
    clamped=False,
    edge_pt_min=5,
    edge_pt_max=40,
)
stiff_analysis.run_buckling_analysis(sigma=10.0, num_eig=20, write_soln=True)
stiff_analysis.post_analysis()