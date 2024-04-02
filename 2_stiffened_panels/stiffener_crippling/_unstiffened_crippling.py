"""
Sean Engelstad, Feb 2024
GT SMDO Lab
"""
import ml_buckling as mlb
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD

flat_plate = mlb.UnstiffenedPlateAnalysis.solvay5320(
    comm=comm,
    bdf_file="plate.bdf",
    a=1.0,
    b=1.0,
    h=0.005,
    ply_angle=0,
)

flat_plate.generate_tripping_bdf(
    nx=30,
    ny=30,
    exx=flat_plate.affine_exx,
    eyy=0.0,
    exy=0.0, # flat_plate.affine_exy,
)

# avg_stresses = flat_plate.run_static_analysis(write_soln=True)

tacs_eigvals, errors = flat_plate.run_buckling_analysis(
    sigma=10.0, num_eig=12, write_soln=True
)