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
parent_parser.add_argument("--clear", type=bool, default=False)

args = parent_parser.parse_args()

assert args.load in ["Nx", "Nxy"]

cpath = os.path.dirname(__file__)
raw_data_folder = os.path.join(cpath, "raw_data")
if not os.path.exists(raw_data_folder) and comm.rank == 0:
    os.mkdir(raw_data_folder)
data_folder = os.path.join(cpath, "data")
if not os.path.exists(data_folder) and comm.rank == 0:
    os.mkdir(data_folder)

stiffened_csv = os.path.join(raw_data_folder, args.load + "_stiffened.csv")
stiff_df = pd.read_csv(stiffened_csv)

X = stiff_df[["rho_0", "xi", "gamma", "zeta", "lambda_star"]].to_numpy()
pred_type = stiff_df["pred_type"].to_numpy()
pred_mask = pred_type == "global"
X = X[pred_mask,:]
# convert all to log scale
X = np.log(X)
Y_stiff = X[:,4:]
X_stiff = X[:,:4]

unstiffened_csv = os.path.join(data_folder, args.load + "_unstiffened.csv")
unstiff_df = pd.read_csv(unstiffened_csv, skiprows=1)

X_unstiff = unstiff_df[["x0", "x1", "x2"]].to_numpy()
n_unstiff = X_unstiff.shape[0]
Y_unstiff = unstiff_df["y"].to_numpy().reshape((n_unstiff,1))

# add gamma=0 parameter to the unstiff data
X_unstiff = np.concatenate([X_unstiff[:,:2], np.zeros((n_unstiff,1)), X_unstiff[:,2:]], axis=1)

# combine the unstiff and stiff data
X_combined = np.concatenate([X_unstiff, X_stiff], axis=0)
Y_combined = np.concatenate([Y_unstiff, Y_stiff], axis=0)

# make a new dataframe and csv file in the reg data folder
new_df_dict = {
    "log(rho_0)" : list(X_combined[:,0]),
    "log(xi)" : list(X_combined[:,1]),
    "log(gamma)" : list(X_combined[:,2]),
    "log(zeta)" : list(X_combined[:,3]),
    "log(lam_star)" : list(Y_combined[:,0]),
}
df = pd.DataFrame(new_df_dict)
df.to_csv("data/Nx_stiffened.csv")