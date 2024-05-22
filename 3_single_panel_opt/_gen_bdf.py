"""
Sean Engelstad, May 2024
GT SMDO Lab

Generate the BDF file with displacement constraints (combined axial + shear).
"""
import ml_buckling as mlb
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD

AR = 1.5
flat_plate = mlb.UnstiffenedPlateAnalysis.solvay5320(
    comm=comm,
    bdf_file="_plate.bdf",
    a=0.1 * AR,
    b=0.1,
    h=0.005,
    ply_angle=45, # so that D11 = D22 and rho0 = rho
)

flat_plate.generate_bdf(
    nx=int(30*AR),
    ny=30,
    exx=1e-3,
    eyy=0.0,
    exy=1e-3,
    clamped=False,
)