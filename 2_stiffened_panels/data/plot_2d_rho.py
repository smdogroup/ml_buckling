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
parent_parser.add_argument("--version", type=int, default=1)
args = parent_parser.parse_args()

# random seed (fixed for now..)
np.random.seed(123456)

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
if args.version == 0:
    suffix = ""
elif args.version == 1:
    suffix = "_smoothed"
# suffix = ""

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

xi_bin = [0.6, 0.8]
zeta_bin = [0.0, 0.5]

ngam = 6
gam_vec = np.geomspace(0 + 1, 15 + 1, ngam+1) - 1
log_gam_vec = np.log(1 + gam_vec)
loggam_bins = [[log_gam_vec[i], log_gam_vec[i+1]] for i in range(ngam)]

xi_mask = np.logical_and(xi_bin[0] <= X[:,0], X[:,0] <= xi_bin[1])
zeta_mask = np.logical_and(zeta_bin[0] <= X[:,2], X[:,2] <= zeta_bin[1])

colors = ["#264653", "#2a9d8f", "#8ab17d", "#e9c46a", "#f4a261", "#e76f51"]
colors = colors[::-1]

for igam,loggam_bin in enumerate(loggam_bins):
    gam_bin = [np.exp(loggam_bin[0])-1, np.exp(loggam_bin[1]) - 1]
    print(f"{gam_bin=} {loggam_bin=}")
    loggam_mask = np.logical_and(loggam_bin[0] <= X[:,-1], X[:,-1] <= loggam_bin[1])

    # print(f"{}")

    full_mask = np.logical_and(xi_mask, zeta_mask)
    full_mask = np.logical_and(full_mask, loggam_mask)

    X_in_range = X[full_mask, :]
    Y_in_range = Y[full_mask, :]

    log_rho = X_in_range[:,1]
    #a, b = best_fit(log_rho, Y_in_range[:])

    print(f"{np.sum(full_mask)=}")

    #if igam == 4:
        # print out dataset to terminal
    arr = np.concatenate([X_in_range, Y_in_range], axis=1)
    print(f"{igam=} {arr=}")

    plt.scatter(
        np.exp(X_in_range[:, 1]),
        np.exp(Y_in_range[:]),
        s=40,
        color=colors[igam % 6],
        edgecolors="black",
        zorder=2 + igam,
        label=r"$\gamma \in $" + f"[{gam_bin[0]:.2f},{gam_bin[1]:.2f}]"
    )

plt.xscale('log')
plt.yscale('log')

    # now also plot the line of best fit
    #Y_fit = a + b * log_rho
    # plt.plot(log_rho, Y_fit, "--", color=colors[igam % 6], zorder=1)

plt.legend()
plt.show()
