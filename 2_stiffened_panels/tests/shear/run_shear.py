import numpy as np
import ml_buckling as mlb
import pandas as pd
from mpi4py import MPI
import os, argparse, sys, argparse

# parent_parser = argparse.ArgumentParser(add_help=False)
# parent_parser.add_argument("--rho0", type=float)
# parent_parser.add_argument("--gamma", type=float)
# args = parent_parser.parse_args()

sys.path.append("../")

from _utils import get_metal_buckling_load
import warnings

comm = MPI.COMM_WORLD

# ----------------------------------
# input_deck = [rho0, xi, gamma, log10(zeta), Ns]
# input_deck = [0.167, 0.589, 5.138, -3.804, 9]

# these two have almost the exact same eigenvalues..
# input_deck = [0.167, 0.589, 5.138, -3.804, 9] 
# input_deck = [0.206, 0.588, 5.85, -4.144, 9]
# input_deck = [0.167, 0.589, 5.138, -3.804, 11] # analysis doesn't run well, needs more stiffeners here

input_deck = [0.135, 0.932, 3.01, -4.5, 11]

# ----------------------------------

zeta = 10.0**input_deck[3]

eig_CF, eig_FEA = get_metal_buckling_load(
    comm,
    rho0=input_deck[0],
    gamma=input_deck[2],
    num_stiff=input_deck[-1],
    sigma_eig=10.0,
    stiff_AR=15.0,
    nelems=2000,
    plate_slenderness=zeta**(-0.5) * 1.5, # tends to be a bit higher SR
    is_axial=False,
    ant_plot_debug=False,
)

if comm.rank == 0:
    print(f"{eig_CF=} {eig_FEA=}")