"""
Sean Engelstad, Feb 2024
GT SMDO Lab
NOTE : copy u*iHat+v*jHat+w*kHat for paraview
"""
import ml_buckling as mlb
import numpy as np
from mpi4py import MPI
import argparse

parent_parser = argparse.ArgumentParser(add_help=False)
parent_parser.add_argument("--load", type=str)
parent_parser.add_argument("--BC", type=str)
parent_parser.add_argument("--AR", type=float, default=1.0)

args = parent_parser.parse_args()

assert args.load in ["Nx", "Nxy", "axial", "shear"]
assert args.BC in ["SS", "CL"]

comm = MPI.COMM_WORLD

flat_plate = mlb.UnstiffenedPlateAnalysis.solvay5320(
    comm=comm,
    bdf_file="plate.bdf",
    a=args.AR * 0.1,
    b=0.1,
    h=0.005,
    ply_angle=45,  # so that D11 = D22 and rho0 = rho
)

nelems = 1000
nmin = int(np.sqrt(nelems / args.AR))
nmax = int(nmin * args.AR)
if args.AR > 1.0:
    nx = nmax
    ny = nmin
else:
    nx = nmin
    ny = nmax

flat_plate.generate_bdf(
    nx=nx,
    ny=ny,
    exx=flat_plate.affine_exx if args.load == "Nx" else 0.0,
    eyy=0.0,
    exy=flat_plate.affine_exy if args.load == "Nxy" else 0.0,
    clamped=args.BC == "CL",
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
