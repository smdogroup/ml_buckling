"""
Sean Engelstad
: 
Demo analysis of a stiffened panel
NOTE : copy u*iHat+v*jHat+w*kHat for paraview
"""

import ml_buckling as mlb
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD

# I think what happens when h becomes very low for large a,b the norm of K is too small and that affects the soln
# to prevent low thickness problems => just make overall plate size smaller

"""
Need to make the stiffener stronger w.r.t. the plate
To exaggarate: have really thick stiffener (non-physically thick)
"""

# the first couple modes are beam-column buckling along the y-direction
# then the 4th and 5th modes are local buckling!

geometry = mlb.StiffenedPlateGeometry(
    a=1.0,
    b=1.0,
    h=1e-2,
    num_stiff=3,
    h_w=7e-2, #1e-1 worked here
    t_w=4e-3, #1e-2 but then crippling happened
)

material = mlb.CompositeMaterial.solvay5320(ref_axis=np.array([1, 0, 0]))

stiff_analysis = mlb.StiffenedPlateAnalysis(
    comm=comm,
    geometry=geometry,
    stiffener_material=material,
    plate_material=material,
)

stiff_analysis.pre_analysis(
    global_mesh_size=0.03,
    exx=stiff_analysis.affine_exx,
    exy=0.0, # stiff_analysis.affine_exy, #0.0
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
