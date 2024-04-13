"""
Sean Engelstad
April 2024, GT SMDO Lab
Goal is to generate a dataset for pure uniaxial and pure shear compression for stiffener crippling data.
Simply supported only and stiffener crippling modes are rejected.

NOTE : copy u*iHat+v*jHat+w*kHat for paraview
"""

import ml_buckling as mlb
import pandas as pd
import numpy as np, os, argparse
from mpi4py import MPI

comm = MPI.COMM_WORLD

# argparse
parent_parser = argparse.ArgumentParser(add_help=False)
parent_parser.add_argument("--load", type=str)
args = parent_parser.parse_args()
assert args.load in ["Nx", "Nxy"]

ply_angle = 0.0
plate_material = mlb.CompositeMaterial.solvay5320(
    ply_angles=[ply_angle], ply_fractions=[1.0], ref_axis=[1, 0, 0]
)
stiff_material = plate_material

# geometry
SR = 200.0
h = 1.0
b = h * SR
SHR = 0.01
num_stiff = 1
# AR_vec = [0.2, 0.3, 0.5, 0.8, 1.0, 1.2, 1.4, 1.6]
AR = 0.5
a = AR * b

# use stiffener height ratio to determine the stiffener height
h_w = a * SHR
stiff_AR = 1.0
t_w = h_w / stiff_AR

geometry = mlb.StiffenedPlateGeometry(
    a=a,
    b=b,
    h=h,
    num_stiff=num_stiff,
    h_w=h_w,
    t_w=t_w,
)

stiffened_plate = mlb.StiffenedPlateAnalysis(
    comm=comm,
    geometry=geometry,
    stiffener_material=stiff_material,
    plate_material=plate_material,
    name="mc",
)

# choose a number of elements in each direction
_nelems = 4000
N = geometry.num_local
AR_s = geometry.a / geometry.h_w
#print(f"AR = {AR}, AR_s = {AR_s}")
nx = np.ceil(np.sqrt(_nelems / (1.0/AR + (N-1) / AR_s)))
ny = np.ceil(nx / AR / N)
nz = max(np.ceil(nx / AR_s),5)

#check_nelems = N * nx * ny + (N-1) * nx * nz
#print(f"check nelems = {check_nelems}")

stiffened_plate.pre_analysis(
    exx=stiffened_plate.affine_exx
    if args.load == "Nx"
    else 0.0,
    exy=stiffened_plate.affine_exy
    if args.load == "Nxy"
    else 0.0,
    clamped=False,
    nx_plate=int(nx),
    ny_plate=int(ny),
    nz_stiff=int(nz),
    # global_mesh_size=global_mesh_size,
    # edge_pt_min=5,
    # edge_pt_max=50,
    _make_rbe=False,  # True
)

print(stiffened_plate)
# exit()

lam_min, mode_type = stiffened_plate.predict_crit_load(
    exx=stiffened_plate.affine_exx
    if args.load == "Nx"
    else 0.0,
    exy=stiffened_plate.affine_exy
    if args.load == "Nxy"
    else 0.0,
    output_global=True,
)
lam_min2 = stiffened_plate.predict_crit_load_old(
    exx=stiffened_plate.affine_exx
    if args.load == "Nx"
    else 0.0,
    exy=stiffened_plate.affine_exy
    if args.load == "Nxy"
    else 0.0,
    output_global=True,
)

tacs_eigvals, errors = stiffened_plate.run_buckling_analysis(
    sigma=5.0, num_eig=50, write_soln=True
)
stiffened_plate.post_analysis()

if comm.rank == 0:
    stiffened_plate.print_mode_classification()

# if abs(errors[0]) > 1e-7: continue

global_lambda_star = stiffened_plate.min_global_mode_eigenvalue

# save data to csv file otherwise because this data point is good
# record the model parameters
data_dict = {
    # training parameter section
    "rho_0": [stiffened_plate.affine_aspect_ratio],
    "xi": [stiffened_plate.xi_plate],
    "gamma": [stiffened_plate.gamma],
    "zeta": [stiffened_plate.zeta_plate],
    "lambda_star": [np.real(global_lambda_star)],
    "pred_lam": [lam_min],
    "pred_type" : [mode_type],
    "pred_lam_old" : [lam_min2],
}

# add raw data section
data_dict["material"] = [plate_material.material_name]
data_dict["ply_angle"] = [ply_angle]
data_dict["SR"] = [SR]
data_dict["AR"] = [AR]
data_dict["h"] = [h]
data_dict["SHR"] = [SHR]
data_dict["SAR"] = [stiff_AR]
data_dict["delta"] = [stiffened_plate.delta]
data_dict["n_stiff"] = [num_stiff]

print(f"data dict = {data_dict}")
