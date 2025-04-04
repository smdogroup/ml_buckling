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
parent_parser.add_argument("--load", type=str, default="Nx")
parent_parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=False, help="Enable or disable axial mode (default: False)")
args = parent_parser.parse_args()

# random seed (fixed for now..)
np.random.seed(123456)

# make folders
# ------------
if not os.path.exists("output"):
    os.mkdir("output")

axial_str = "axial" if args.load == "Nx" else "shear"
base_name = f"{axial_str}"

# get the dataset
# ---------------

csv_filename = f"{args.load}_stiffened"
df = pd.read_csv(csv_filename + ".csv")

# extract only the model columns
X0 = df[["log(1+xi)", "log(rho_0)", "log(1+10^3*zeta)", "log(1+gamma)"]].to_numpy()
Y = df["log(eig_FEA)"].to_numpy()
Y = np.reshape(Y, newshape=(Y.shape[0], 1))

# affine transform
use_affine = args.load == "Nx" # only use for axial
# use_affine = False
if use_affine:
    X = affine_transform(X0, is_log=True)
else: 
    X = X0

def best_fit(X, Y):
    # get the line of best fit

    xbar = sum(X)/len(X)
    ybar = sum(Y)/len(Y)
    n = len(X) # or len(Y)

    numer = sum([xi*yi for xi,yi in zip(X, Y)]) - n * xbar * ybar
    denum = sum([xi**2 for xi in X]) - n * xbar**2

    b = numer / denum
    a = ybar - b * xbar

    # print(f'best fit line:\ny = {:.2f} + {:.2f}x'.format(a, b))

    return a, b

# plot 2d gamma plots
# -------------------

xi_bin = [0.5, 0.7]
zeta_bin = [0.2, 0.4]

nrho = 8
rho_vec = np.geomspace(0.2, 10.0, nrho+1)
log_rho_vec = np.log(rho_vec)
logrho_bins = [[log_rho_vec[i], log_rho_vec[i+1]] for i in range(nrho)]

xi_mask = np.logical_and(xi_bin[0] <= X[:,0], X[:,0] <= xi_bin[1])
zeta_mask = np.logical_and(zeta_bin[0] <= X[:,2], X[:,2] <= zeta_bin[1])

colors = ["#264653", "#2a9d8f", "#8ab17d", "#e9c46a", "#f4a261", "#e76f51"]
colors = colors[::-1]

for irho,logrho_bin in enumerate(logrho_bins):
    logrho_mask = np.logical_and(logrho_bin[0] <= X[:,1], X[:,1] <= logrho_bin[1])

    # print(f"{}")

    full_mask = np.logical_and(xi_mask, zeta_mask)
    full_mask = np.logical_and(full_mask, logrho_mask)

    X_in_range = X[full_mask, :]
    Y_in_range = Y[full_mask, :]

    log_gamma = X_in_range[:,-1]
    a, b = best_fit(log_gamma, Y_in_range[:])

    print(f"{np.sum(full_mask)=}")

    plt.scatter(
        X_in_range[:, -1],
        Y_in_range[:],
        s=20,
        color=colors[irho % 6],
        edgecolors="black",
        zorder=2 + irho,
        label=f"logrho-{irho}"
    )

    # now also plot the line of best fit
    Y_fit = a + b * log_gamma
    plt.plot(log_gamma, Y_fit, "--", color=colors[irho % 6], zorder=1)

plt.legend()
plt.show()