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
parent_parser.add_argument("--load", type=str, default="Nx")
parent_parser.add_argument(
    "--clear", default=False, action=argparse.BooleanOptionalAction
)

args = parent_parser.parse_args()

assert args.load in ["Nx", "Nxy"]

cpath = os.path.dirname(__file__)
# raw_data_folder = os.path.join(cpath, "raw_data")
# if not os.path.exists(raw_data_folder) and comm.rank == 0:
#     os.mkdir(raw_data_folder)
data_folder = os.path.join(cpath, "data")
if not os.path.exists(data_folder) and comm.rank == 0:
    os.mkdir(data_folder)

stiffened_csv = args.load + "_raw_stiffened.csv"
stiff_df = pd.read_csv(stiffened_csv)

X = stiff_df[["xi", "rho_0", "log10(zeta)", "gamma", "eig_FEA"]].to_numpy()

# convert xi to log(1+xi)
# X[:, 0] = np.log(1.0 + X[:, 0])
# # convert rho_0 to log(rho_0)
# X[:, 1] = np.log(X[:, 1])
# # convert zeta to log(1+10^3*zeta)
# zeta = np.power(10.0, X[:,2])
# X[:, 2] = np.log(1.0 + 1e3 * zeta)
# # convert gamma to log(1+gamma)
# X[:, 3] = np.log(1.0 + X[:, 3])
# # convert eig_FEA to log(eig_FEA)
# X[:, 4] = np.log(X[:, 4])

# remove nan
not_nan_mask = np.logical_not(np.isnan(X[:,4]))

Y_stiff = X[not_nan_mask, 4:5]
X_stiff = X[not_nan_mask, :4]

Y_CF = stiff_df[["eig_CF"]].to_numpy()[not_nan_mask,:]
FEA_CF_ratio = Y_stiff[:,0] / Y_CF[:,0]

X2 = stiff_df[["num_stiff", "material"]].to_numpy()

# make a new dataframe and csv file in the reg data folder
new_df_dict = {
    "xi": list(X_stiff[:, 0]),
    "rho_0": list(X_stiff[:, 1]),
    "log10(zeta)": list(X_stiff[:, 2]),
    "gamma": list(X_stiff[:, 3]),
    "num_stiff": list(X2[not_nan_mask,0]),
    "material": list(X2[not_nan_mask,1]),
    "eig_FEA": list(Y_stiff[:, 0]),
    "eig_CF": list(Y_CF[:,0]),
    "FEA/CF" : list(FEA_CF_ratio),
}
df = pd.DataFrame(new_df_dict)
df.to_csv(f"{args.load}_raw_stiffened_analyze.csv", float_format="%.6f")
