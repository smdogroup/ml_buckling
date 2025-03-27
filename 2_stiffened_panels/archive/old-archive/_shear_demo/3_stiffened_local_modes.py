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
NOTE : copy u*iHat+v*jHat+w*kHat for viewing
"""

# the first couple modes are beam-column buckling along the y-direction
# then the 4th and 5th modes are local buckling!

geometry = mlb.StiffenedPlateGeometry(
    a=0.1,
    b=0.1,
    h=3e-3,
    num_stiff=3,
    w_b=6e-3,
    t_b=3e-3,
    h_w=1e-2,
    t_w=27
    * 8e-2,  # if the wall thickness is too low => stiffener crimping failure happens
)

# TODO : still getting funky results with stiffener crippled modes for the shear case

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
tacs_eigvals, errors = stiff_analysis.run_buckling_analysis(
    sigma=10.0, num_eig=20, write_soln=True
)
stiff_analysis.post_analysis()

print(f"tacs eigvals = {tacs_eigvals}")
print(f"errors = {errors}")
