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

stiffened_csv = os.path.join(data_folder, args.load + "_raw_stiffened.csv")
stiff_df = pd.read_csv(stiffened_csv)

X = stiff_df[["xi", "rho_0", "zeta", "gamma", "eig_FEA"]].to_numpy()

# convert xi to log(1+xi)
X[:, 0] = np.log(1.0 + X[:, 0])
# convert rho_0 to log(rho_0)
X[:, 1] = np.log(X[:, 1])
# convert zeta to log(1+10^3*zeta)
X[:, 2] = np.log(1.0 + 1e3 * X[:, 2])
# convert gamma to log(1+gamma)
X[:, 3] = np.log(1.0 + X[:, 3])
# convert eig_FEA to log(eig_FEA)
X[:, 4] = np.log(X[:, 4])

Y_stiff = X[:, 4:5]
X_stiff = X[:, :4]

unstiffened_csv = os.path.join(data_folder, args.load + "_unstiffened.csv")
unstiff_df = pd.read_csv(unstiffened_csv)  # , skiprows=1)

# print(f"unstiff df = {unstiff_df}")

X_unstiff = unstiff_df[["x0", "x1", "x2"]].to_numpy()
n_unstiff = X_unstiff.shape[0]
Y_unstiff = unstiff_df["y"].to_numpy().reshape((n_unstiff, 1))

# add gamma=0 parameter to the unstiff data
# assume gamma=0 is approx exp(-6) => -6 log scale
X_unstiff = np.concatenate([X_unstiff[:, :], np.zeros((n_unstiff, 1))], axis=1)

# combine the unstiff and stiff data
X_combined = np.concatenate([X_unstiff, X_stiff], axis=0)
Y_combined = np.concatenate([Y_unstiff, Y_stiff], axis=0)

# make a new dataframe and csv file in the reg data folder
new_df_dict = {
    "log(1+xi)": list(X_combined[:, 0]),
    "log(rho_0)": list(X_combined[:, 1]),
    "log(1+10^3*zeta)": list(X_combined[:, 2]),
    "log(1+gamma)": list(X_combined[:, 3]),
    "log(eig_FEA)": list(Y_combined[:, 0]),
}
df = pd.DataFrame(new_df_dict)
df.to_csv(f"data/{args.load}_stiffened.csv")
