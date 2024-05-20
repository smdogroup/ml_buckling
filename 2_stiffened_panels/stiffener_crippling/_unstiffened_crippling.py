"""
Sean Engelstad, Feb 2024
GT SMDO Lab
"""
import ml_buckling as mlb
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD

h = 1.0
AR = 8.0
SR = 200.0
b = h * SR
a = b * AR

flat_plate = mlb.UnstiffenedPlateAnalysis.solvay5320(
    comm=comm,
    bdf_file="plate.bdf",
    a=a,
    b=b,
    h=h,
    ply_angle=0,
)

load_factor = 0.03
flat_plate.generate_tripping_bdf(
    nx=100,
    ny=20,
    exx=flat_plate.affine_exx * load_factor,
    eyy=0.0,
    exy=0.0,  # flat_plate.affine_exy,
)

print(f"xi = {flat_plate.Dstar}")
epsilon = flat_plate.generalized_poisson
print(f"eps = {epsilon}")
# exit()

avg_in_plane_loads = flat_plate.run_static_analysis(write_soln=True)
# exit()

tacs_eigvals, errors = flat_plate.run_buckling_analysis(
    sigma=5.0, num_eig=12, write_soln=True
)

# compare to exact eigenvalue
tacs_eigval = tacs_eigvals[0] * load_factor
CF_eigval = 0.476 * flat_plate.xi
rel_err = (tacs_eigval - CF_eigval) / CF_eigval

print(f"tacs eigval = {tacs_eigval}")
print(f"CF eigval = {CF_eigval}")
print(f"rel err = {rel_err}")
