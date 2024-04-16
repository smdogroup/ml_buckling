import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import niceplots, scipy, time, os
import argparse
from mpl_toolkits import mplot3d
from matplotlib import cm
import shutil

"""
This time I'll try a Gaussian Process model to fit the axial critical load surrogate model
Inputs: D*, a0/b0, ln(b/h)
Output: k_x0
"""
# parse the arguments
parent_parser = argparse.ArgumentParser(add_help=False)
parent_parser.add_argument("--load", type=str)

args = parent_parser.parse_args()

assert args.load in ["Nx", "Nxy"]

load = args.load

# load the Nxcrit dataset
load_prefix = "Nx_stiffened" if load == "Nx" else "Nxy"
csv_filename = f"{load}_stiffened"
df = pd.read_csv("data/" + csv_filename + ".csv")

# extract only the model columns
# TODO : if need more inputs => could maybe try adding log(E11/E22) in as a parameter?
# or also log(E11/G12)
X = df[["log(rho_0)", "log(xi)", "log(zeta)", "log(gamma)"]].to_numpy()
Y = df["log(lam_star)"].to_numpy()
Y = np.reshape(Y, newshape=(Y.shape[0], 1))

print(f"Monte Carlo #data = {X.shape[0]}")
N_data = X.shape[0]

n_train = int(0.9 * N_data)

# REMOVE THE OUTLIERS in local 4d regions
# loop over different slenderness bins
zeta_bins = [
    [10.0, 20.0],
    [20.0, 50.0],
    [50.0, 100.0],
    [100.0, 200.0],
]
xi_bins = [[0.25 * i, 0.25 * (i + 1)] for i in range(7)]
log_xi_bins = [list(np.log(np.array(xi_bin))) for xi_bin in xi_bins]
# added smaller and larger bins here cause we miss some of the outliers near the higher a0/b0 with less data
aff_AR_bins = (
    [[0.5 * i, 0.5 * (i + 1)] for i in range(4)]
    + [[1.0 * i, 1.0 * (i + 1)] for i in range(2, 5)]
    + [[2.0, 10.0]]
)

_plot = True
_plot_Dstar_2d = False
_plot_slender_2d = False
_plot_3d = True
_plot_model_fit = True
_plot_model_fit_xi = True

# make a folder for the model fitting
plots_folder = os.path.join(os.getcwd(), "plots")
sub_plots_folder = os.path.join(plots_folder, csv_filename)
GP_folder = os.path.join(sub_plots_folder, "GP")
for ifolder, folder in enumerate(
    [
        plots_folder,
        sub_plots_folder,
        GP_folder,
    ]
):
    if ifolder > 0 and os.path.exists(folder):
        shutil.rmtree(folder)
    if not os.path.exists(folder):
        os.mkdir(folder)

plt.style.use(niceplots.get_style())
xi = X[:, 0]
affine_AR = X[:, 1]
zeta = X[:, 2]
gamma = X[:, 3]
lam_star = Y[:, 0]

n_data = xi.shape[0]

# split into training and test datasets
n_total = X.shape[0]
n_test = n_total - n_train
assert n_test > 100

# reorder the data
indices = [_ for _ in range(n_total)]
train_indices = np.random.choice(indices, size=n_train)
test_indices = [_ for _ in range(n_total) if not(_ in train_indices)]

X_train = X[train_indices, :]
X_test = X[test_indices, :]
Y_train = Y[train_indices, :]
Y_test = Y[test_indices, :]

# TRAIN THE MODEL HYPERPARAMETERS
# by maximizing the log marginal likelihood log p(y|X)
# or minimizing the negative log marginal likelihood
# ----------------------------------------------------

# y is the training set observations
y = Y_train

# update the local hyperparameter variables
# initial hyperparameter vector
# sigma_n, sigma_f, L1, L2, L3
theta0 = np.array([1e-1, 3e-1, -1, 0.2, 1.0, 1.0, 0.3, 2, 1.0])
sigma_n = 1e-2


def relu(x):
    return max([0.0, x])

def soft_relu(x, rho=10):
    return 1.0 / rho * np.log(1 + np.exp(rho * x))


def kernel(xp, xq, theta):
    # xp, xq are Nx1,Mx1 vectors (ln(xi), ln(rho_0), ln(zeta), ln(gamma))
    vec = xp - xq

    S1 = theta[0]
    S2 = theta[1]
    c = theta[2]
    L1 = theta[3]
    S4 = theta[4]
    S5 = theta[5]
    L2 = theta[6]
    alpha_1 = theta[7]
    L3 = theta[8]

    d1 = vec[1]  # first two entries
    d2 = vec[2]
    d3 = vec[3]

    # log(xi) direction
    kernel0 = S1 ** 2 + S2 ** 2 * soft_relu(xp[0] - c, alpha_1) * soft_relu(
        xq[0] - c, alpha_1
    )
    # log(rho_0) direction
    kernel1 = (
        np.exp(-0.5 * (d1 ** 2 / L1 ** 2))
        * soft_relu(1 - abs(xp[1]))
        * soft_relu(1 - abs(xq[1]))
        + S4
        + S5 * soft_relu(-xp[1]) * soft_relu(-xq[1])
    )
    # log(zeta) direction
    kernel2 = np.exp(-0.5 * d2 ** 2 / L2 ** 2)
    # log(gamma) direction
    kernel3 = np.exp(-0.5 * d3 **2 / L3 ** 2)
    return kernel0 * kernel1 * kernel2 * kernel3


# compute the training kernel matrix
K_y = np.array(
    [
        [kernel(X_train[i, :], X_train[j, :], theta0) for i in range(n_train)]
        for j in range(n_train)
    ]
) + sigma_n ** 2 * np.eye(n_train)

# compute the objective : maximize log marginal likelihood
# sign,log_detK = np.linalg.slogdet(K_y) # special numpy routine for log(det(K))
# print(f"\tlog detK = {log_detK}, sign = {sign}")
# _start = time.time()
alpha = np.linalg.solve(K_y, y)

# plot the model and some of the data near the model range in D*=1, AR from 0.5 to 5.0, b/h=100
# ---------------------------------------------------------------------------------------------

if _plot:
    # get the available colors
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

    plt.style.use(niceplots.get_style())

    