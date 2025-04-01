import sys
import pandas as pd
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt

sys.path.append("../src/")
from kernel_library import *
from plot_utils import plot_3d_gamma, plot_3d_xi
from data_transforms import affine_transform

# argparse
# --------

parent_parser = argparse.ArgumentParser(add_help=False)
parent_parser.add_argument(
    "--axial", default=False, action=argparse.BooleanOptionalAction
)
args = parent_parser.parse_args()

# random seed (fixed for now..)
np.random.seed(123456)

# make folders
# ------------

axial_str = "axial" if args.axial else "shear"
base_name = f"{axial_str}"

# get the dataset
# ---------------

load_str = "Nx" if args.axial else "Nxy"
df = pd.read_csv(f"{load_str}_raw_stiffened.csv")

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

