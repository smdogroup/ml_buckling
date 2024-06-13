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
    h_w=3e-3,
    t_w=1e-3,  
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
    ny_plate=60,
    nz_stiff=10,
    exx=stiff_analysis.affine_exx,
    exy=0.0,    #  stiff_analysis.affine_exy,     #0.0
    clamped=False,
)

if comm.rank == 0:
    print(stiff_analysis)

# predict closed-form solution
lam_ND,mode_type = stiff_analysis.predict_crit_load(exx=stiff_analysis.affine_exx)
tacs_eigvals, errors = stiff_analysis.run_buckling_analysis(
    sigma=10.0, num_eig=20, write_soln=True
)
stiff_analysis.post_analysis()

if comm.rank == 0:
    print(f"tacs eigvals = {tacs_eigvals}")
    print(f"errors = {errors}")

# compare actual eigval to predicted one:
if comm.rank == 0:
    print(f"fea eigval = {tacs_eigvals[0]}")
    print(f"CF eigval = {lam_ND}")
