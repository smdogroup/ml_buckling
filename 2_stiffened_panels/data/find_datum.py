import sys
import pandas as pd
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt

sys.path.append("../src/")
from kernel_library import *
# from plot_utils import plot_3d_gamma, plot_3d_xi
from data_transforms import affine_transform

# argparse
# --------

parent_parser = argparse.ArgumentParser(add_help=False)
parent_parser.add_argument(
    "--axial", default=False, action=argparse.BooleanOptionalAction
)
# parent_parser.add_argument("--version", type=int, default=0)
args = parent_parser.parse_args()

# ---------------------------------------

gamma_bin_ind = 3
rho0_approx = 0.135
rho0_bin = [rho0_approx*0.9, rho0_approx*1.1]

# ---------------------------------------


# make folders
# ------------
if not os.path.exists("output"):
    os.mkdir("output")

axial_str = "axial" if args.axial else "shear"
base_name = f"{axial_str}"

# get the dataset
# ---------------

load_str = "Nx" if args.axial else "Nxy"
csv_filename = f"{load_str}_raw_stiffened"
suffix = None
# if args.version == 0:
#     suffix = ""
# elif args.version > 0:
#     suffix = "_v" + str(args.version)
suffix = ""

df = pd.read_csv(csv_filename + suffix + ".csv")

# extract only the model columns
X0 = df[["xi", "rho_0", "log10(zeta)", "gamma"]].to_numpy()
Y = df["eig_FEA"].to_numpy()
Y = np.reshape(Y, newshape=(Y.shape[0], 1))

# transform
X0[:,0] = np.log(1.0 + X0[:,0])
X0[:,1] = np.log(X0[:,1])
X0[:,2] = np.log(1.0 + 1e3 * np.power(10.0, X0[:,2]))
X0[:,3] = np.log(1 + X0[:,3])
Y = np.log(Y)

# affine transform
use_affine = args.axial # only use for axial
# use_affine = False
if use_affine:
    X = affine_transform(X0, is_log=True)
else: 
    X = X0

# plot 2d gamma plots
# -------------------

xi_bin = [0.6, 0.8]
zeta_bin = [0.0, 0.5]

ngam = 6
gam_vec = np.geomspace(0 + 1, 15 + 1, ngam+1) - 1
log_gam_vec = np.log(1 + gam_vec)
loggam_bins = [[log_gam_vec[i], log_gam_vec[i+1]] for i in range(ngam)]

xi_mask = np.logical_and(xi_bin[0] <= X[:,0], X[:,0] <= xi_bin[1])
zeta_mask = np.logical_and(zeta_bin[0] <= X[:,2], X[:,2] <= zeta_bin[1])

loggam_bin = loggam_bins[gamma_bin_ind]
loggam_mask = np.logical_and(loggam_bin[0] <= X[:,-1], X[:,-1] <= loggam_bin[1])
full_mask = np.logical_and(xi_mask, zeta_mask)
full_mask = np.logical_and(full_mask, loggam_mask)
print(f"{rho0_bin=}")
rho0_mask = np.logical_and(X[:,1] < np.log(rho0_bin[1]), X[:,1] > np.log(rho0_bin[0]))
full_mask = np.logical_and(full_mask, rho0_mask)

indices = np.array([_+2 for _ in range(X.shape[0])])
myind = indices[full_mask]
print(f"{myind=}")

# expected eigenvalues
Xred = X[full_mask,:]
gamma = np.exp(Xred[:,-1]) - 1.0
rho0 = np.exp(Xred[:,1])
# linearized form of N12cr (1+gamma) / rho0**2.0 here
N12bar_pred = 9.5 * (1.0 + gamma) / rho0**1.65

print(f"{N12bar_pred=}")

# print(f"{np.exp(X[full_mask,1])}")