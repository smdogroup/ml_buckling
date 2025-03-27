"""
Sean Engelstad, Feb 2024
GT SMDO Lab
NOTE : copy u*iHat+v*jHat+w*kHat for paraview
"""
import ml_buckling as mlb
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD

flat_plate = mlb.UnstiffenedPlateAnalysis.solvay5320(
    comm=comm,
    bdf_file="plate.bdf",
    a=0.1,
    b=0.1,
    h=0.005,
    ply_angle=0,
)

flat_plate.generate_bdf(
    nx=30,
    ny=30,
    exx=flat_plate.affine_exx,
    eyy=0.0,
    exy=0.0,
    clamped=False,
)

# avg_stresses = flat_plate.run_static_analysis(write_soln=True)

tacs_eigvals, errors = flat_plate.run_buckling_analysis(
    sigma=10.0, num_eig=15, write_soln=True
)

# in previous monte carlo iteration, this trial got lam1 = 10.54
# whereas here we get lam1 = 2.56 with 1e-14 error in the eigenvalue solver
# seems like the error was bad on that solution? Need a way from python to check error in solution..
if comm.rank == 0:
    print(f"tacs eigvals = {tacs_eigvals}")
    print(f"errors = {errors}")
