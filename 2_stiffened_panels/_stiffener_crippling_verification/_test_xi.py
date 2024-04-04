"""
Sean Engelstad
Feb 2024, GT SMDO Lab
Goal is to generate data for pure uniaxial compression failure in the x-direction
For now just simply supported only..

gen_mc_data.py : generate monte carlo training data for each of the surrogate models
"""

import ml_buckling as mlb
import numpy as np
import pandas as pd
import os
from mpi4py import MPI

comm = MPI.COMM_WORLD

# MODEL INPUTS SECTION
# --------------------------------------------
# --------------------------------------------

# number of random samples (traverse all AR for each of them)
N = 4000

# END OF MODEL INPUTS SECTION
# --------------------------------------------
# --------------------------------------------

# start of computations
data_dict_list = []

# make a folder for the pandas data
cpath = os.path.dirname(__file__)
data_folder = os.path.join(cpath, "data")
if not os.path.exists(data_folder) and comm.rank == 0:
    os.mkdir(data_folder)

# clear the csv file
_clear_data = False

csv_file = os.path.join(data_folder, "stiffener_crippling.csv")
if _clear_data:
    if os.path.exists(csv_file) and comm.rank == 0:
        os.remove(csv_file)

# maybe it would be better to use a materials database and select a random material then
# and use material classmethods instead

# TODO : rewrite this by randomly choose an isotropic / composite material
# randomly choose a b and h pair until AR and affine_AR in range
# this way a lot of random slenderness values will be used
# iterate by sweeping AR through the data

inner_ct = 0

# run the nominal plate for mode tracking
# nominal_plate = mlb.UnstiffenedPlateAnalysis.hexcelIM7(
#     comm=comm,
#     bdf_file="plate.bdf",
#     a=1.0,
#     b=1.0,
#     h=0.01,
#     ply_angle=0,
# )

# nominal_plate.generate_tripping_bdf(
#     nx=30,
#     ny=30,
#     exx=nominal_plate.affine_exx,
#     eyy=0.0,
#     exy=0.0,
# )
# nom_eigvals, _ = nominal_plate.run_buckling_analysis(
#     sigma=5.0, num_eig=20, write_soln=False
# )

log_AR = np.linspace(-1.0, 2.0, 30)
AR_vec = np.power(10.0, log_AR)

for foo in range(N):  # until has generated this many samples
    # randomly generate the material
    materials = mlb.UnstiffenedPlateAnalysis.get_materials()
    material = np.random.choice(np.array(materials))
    ply_angle = np.random.uniform(0.0, 90.0)

    # random geometry, min thickness so that K,G matrices have good norm
    log_slenderness = np.random.uniform(np.log(10.0), np.log(200.0))
    slenderness = np.exp(log_slenderness)
    h = 1.0
    b = (
        h * slenderness
    )  # verified that different b values don't influence non-dim buckling load

    fail_ct = 0

    AR = 5.0
    a = AR * b

    for ply_angle in np.linspace(0.0, 90, 20):
        new_plate: mlb.UnstiffenedPlateAnalysis = material(
            comm,
            bdf_file="plate.bdf",
            a=a,
            b=b,
            h=h,
            ply_angle=ply_angle,
        )
        xi = new_plate.xi
        print(f"mat {new_plate.material_name}, ply angle {ply_angle}, xi {xi}")