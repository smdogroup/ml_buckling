"""
Sean Engelstad
: 
Demo analysis of a stiffened panel
NOTE : copy u*iHat+v*jHat+w*kHat for paraview
"""

import ml_buckling as mlb
from mpi4py import MPI

comm = MPI.COMM_WORLD

# I think what happens when h becomes very low for large a,b the norm of K is too small and that affects the soln
# to prevent low thickness problems => just make overall plate size smaller

# TODO : still getting funky results with stiffener crippled modes for the shear case

geometry = mlb.StiffenedPlateGeometry(
    a=0.1, 
    b=0.1,
    h=5e-3,
    num_stiff=3,
    w_b=6e-3,
    t_b=2e-3,
    h_w=3e-3,
    t_w=1e-3, # if the wall thickness is too low => stiffener crimping failure happens
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
    exx=0.0,
    exy=stiff_analysis.affine_exy,
    clamped=False,
    edge_pt_min=5,
    edge_pt_max=40,
)
tacs_eigvals, errors = stiff_analysis.run_buckling_analysis(sigma=10.0, num_eig=20, write_soln=True)
stiff_analysis.post_analysis()

print(f"tacs eigvals = {tacs_eigvals}")
print(f"errors = {errors}")