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

geometry = mlb.StiffenedPlateGeometry(
    a=0.1,
    b=0.1,
    h=5e-3,
    num_stiff=3,
    h_w=1e-2,
    t_w=1e-3,  # if the wall thickness is too low => stiffener crimping failure happens
)

material = mlb.CompositeMaterial.solvay5320(ref_axis=np.array([1, 0, 0]))

stiff_analysis = mlb.StiffenedPlateAnalysis(
    comm=comm,
    geometry=geometry,
    stiffener_material=material,
    plate_material=material,
)

stiff_analysis.pre_analysis(
    nx_plate=30,
    ny_plate=10,
    nz_stiff=10,
    exx=stiff_analysis.affine_exx,
    exy=0.0,
    clamped=False,
)
tacs_eigvals, errors = stiff_analysis.run_buckling_analysis(
    sigma=10.0, num_eig=20, write_soln=True
)
stiff_analysis.post_analysis()

print(f"tacs eigvals = {tacs_eigvals}")
print(f"errors = {errors}")
