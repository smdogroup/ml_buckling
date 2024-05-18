import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import niceplots, scipy, time, os
import argparse
from mpl_toolkits import mplot3d
from matplotlib import cm
import shutil, random

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
X = df[["log(xi)", "log(rho_0)", "log(1+10^3*zeta)", "log(1+gamma)"]].to_numpy()
Y = df["log(lam_star)"].to_numpy()
Y = np.reshape(Y, newshape=(Y.shape[0], 1))

N_data = X.shape[0]

# n_train = int(0.9 * N_data)
n_train = 1000

print(f"Monte Carlo #data training {n_train} / {X.shape[0]} data points")

# randomly permute the arrays
rand_perm = np.random.permutation(N_data)
X = X[rand_perm, :]
Y = Y[rand_perm, :]

# temporarily remove all gamma >= 0 data
# just show xi in third axis
# gamma_keep_mask = X[:,3] == 0.0
# X = X[gamma_keep_mask,:]
# Y = Y[gamma_keep_mask,:]

# REMOVE THE OUTLIERS in local 4d regions
# loop over different slenderness bins
xi_bins = [[0.25 * i, 0.25 * (i + 1)] for i in range(1,7)]
log_xi_bins = [list(np.log(np.array(xi_bin))) for xi_bin in xi_bins]
log_gamma_bins = [[0,1], [1,2], [2,3], [3,4]]

_plot = True
_plot_gamma = False
_plot_3d = True

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

n_data = X.shape[0]

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
# theta0 = np.array([1e-1, 3e-1, -1, 0.2, 1.0, 1.0, 0.5, 2, 0.8, 1.0, 0.2])
theta0 = [9.99999999e-05, 9.99999997e-05, 1.59340141e-01, 7.26056398e-01,
 9.27506384e-01, 4.98424731e-01, 7.75092936e-01, 9.96708608e-01,
 9.99999998e-05, 9.99999998e-05, 6.96879851e-01, 1.00000000e-04,]
sigma_n = 1e-2 #1e-1 was old value

def relu(x):
    return max([0.0, x])

def soft_relu(x, rho=10):
    return 1.0 / rho * np.log(1 + np.exp(rho * x))


# def kernel(xp, xq, theta):
#     # xp, xq are Nx1,Mx1 vectors (ln(xi), ln(rho_0), ln(zeta), ln(gamma))
#     vec = xp - xq

#     S1 = theta[0]
#     S2 = theta[1]
#     c = theta[2]
#     L1 = theta[3]
#     S4 = theta[4]
#     S5 = theta[5]
#     L2 = theta[6]
#     alpha_1 = theta[7]
#     L3 = theta[8]
#     S6 = theta[9]
#     S7 = theta[10]

#     d1 = vec[1]  # first two entries
#     d2 = vec[2]
#     d3 = vec[3]

#     # log(xi) direction
#     kernel0 = S1 ** 2 + S2 ** 2 * soft_relu(xp[0] - c, alpha_1) * soft_relu(
#         xq[0] - c, alpha_1
#     )
#     # log(rho_0) direction
#     kernel1 = (
#         np.exp(-0.5 * (d1 ** 2 / L1 ** 2))
#         * soft_relu(1 - abs(xp[1]))
#         * soft_relu(1 - abs(xq[1]))
#         + S4
#         + S5 * soft_relu(-xp[1]) * soft_relu(-xq[1])
#     )
#     # log(zeta) direction
#     kernel2 = np.exp(-0.5 * d2 ** 2 / L2 ** 2)
#     # log(gamma) direction
#     kernel3 = S6 * np.exp(-0.5 * d3 **2 / L3 ** 2) + S7 * xp[3] * xq[3]
#     # TODO : should this be + kernel3 or * kernel3 ?
#     #return kernel0 * kernel1 * kernel2 + 2.0 * kernel3
#     return kernel0 * kernel1 * kernel2 * kernel3

def kernel(xp, xq, theta):
    # xp, xq are Nx1,Mx1 vectors (ln(xi), ln(rho_0), ln(1 + 10^3 * zeta), ln(1 + gamma))
    vec = xp - xq

    S1 = theta[0]
    S2 = theta[1]
    L1 = theta[2]
    S4 = theta[3]
    S5 = theta[4]
    L2 = theta[5]
    L3 = theta[6]
    S6 = theta[7]
    S7 = theta[8]
    S8 = theta[9]
    S9 = theta[10]
    # sigma_n = theta[11]

    d1 = vec[1]  # first two entries
    d2 = vec[2]
    d3 = vec[3]

    # log(xi) direction
    kernel0 = S1 ** 2 + S2 ** 2 * xp[0] * xq[0]
    # log(rho_0) direction
    kernel1 = (
        np.exp(-0.5 * (d1 ** 2 / L1 ** 2))
        * soft_relu(1 - abs(xp[1]))
        * soft_relu(1 - abs(xq[1]))
        + S4
        + S5 * soft_relu(-xp[1]) * soft_relu(-xq[1])
    )
    # log(zeta) direction
    kernel2 = S6 * np.exp(-0.5 * d2 ** 2 / L2 ** 2) + S7 * xp[2] * xq[2]
    # log(gamma) direction
    kernel3 = S8 * np.exp(-0.5 * d3 **2 / L3 ** 2) + S9 * xp[3] * xq[3]
    return kernel0 * kernel1 * kernel2 * kernel3

_compute = True
if _compute:
    # compute the training kernel matrix
    K_y = np.array(
        [
            [kernel(X_train[i, :], X_train[j, :], theta0) for i in range(n_train)]
            for j in range(n_train)
        ]
    ) + sigma_n ** 2 * np.eye(n_train)

    alpha = np.linalg.solve(K_y, y)

# plot the model and some of the data near the model range in D*=1, AR from 0.5 to 5.0, b/h=100
# ---------------------------------------------------------------------------------------------

if _plot:
    # get the available colors
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

    plt.style.use(niceplots.get_style())

    if _plot_gamma:
        # 3d plot of rho_0, gamma, lam_star for a particular xi and zeta range
        #xi_bin = [-1.2, -0.8]
        xi_bin = [-0.5, 0.5]
        # xi_bin = [-2.0, 2.0]
        xi_mask = np.logical_and(xi_bin[0] <= X[:,0], X[:,0] <= xi_bin[1])
        avg_xi = 0.0
        #zeta_bin = [6.0, 8.0]
        zeta_bin = [0.0, 1.0]
        # zeta_bin = [0.0, 8.0]
        zeta_mask = np.logical_and(zeta_bin[0] <= X[:,2], X[:,2] <= zeta_bin[1])
        avg_zeta = 0.5
        xi_zeta_mask = np.logical_and(xi_mask, zeta_mask)

        plt.figure(f"3d rho_0, gamma, lam_star")

        colors = plt.cm.jet(np.linspace(0.0, 1.0, len(log_gamma_bins)))

        for igamma,gamma_bin in enumerate(log_gamma_bins):

            gamma_mask = np.logical_and(gamma_bin[0] <= X[:,3], X[:,3] <= gamma_bin[1])
            mask = np.logical_and(xi_zeta_mask, gamma_mask)

            X_in_range = X[mask,:]
            Y_in_range = Y[mask,:]

            # print(f"X in range = {X_in_range}")
            # print(f"Y in range = {Y_in_range}")


            plt.plot(
                X_in_range[:,1],
                Y_in_range[:,0],
                "o",
                color=colors[igamma],
                zorder=1+igamma
            )

        plt.savefig(os.path.join(GP_folder, f"2d-unstiffened.png"), dpi=400)
        plt.close(f"3d rho_0, gamma, lam_star")

    if _plot_3d:

        # 3d plot of rho_0, gamma, lam_star for a particular xi and zeta range
        # xi_bin = [-0.5, 0.5]
        # smaller values of xi have higher gamma
        xi_bin = [-1.15, -0.85]
        # xi_bin = [-2.0, 2.0]
        xi_mask = np.logical_and(xi_bin[0] <= X[:,0], X[:,0] <= xi_bin[1])
        avg_xi = -1.15
        # zeta_bin = [0.0, 8.0]
        zeta_bin = [0.0, 1.0]
        # zeta_bin = [0.0, 8.0]
        zeta_mask = np.logical_and(zeta_bin[0] <= X[:,2], X[:,2] <= zeta_bin[1])
        avg_zeta = 0.5
        xi_zeta_mask = np.logical_and(xi_mask, zeta_mask)

        plt.figure(f"3d rho_0, gamma, lam_star")
        ax = plt.axes(projection="3d", computed_zorder=False)

        colors = plt.cm.jet(np.linspace(0.0, 1.0, len(log_gamma_bins)))

        for igamma,gamma_bin in enumerate(log_gamma_bins):

            gamma_mask = np.logical_and(gamma_bin[0] <= X[:,3], X[:,3] <= gamma_bin[1])
            mask = np.logical_and(xi_zeta_mask, gamma_mask)

            X_in_range = X[mask,:]
            Y_in_range = Y[mask,:]

            #print(f"X in range = {X_in_range}")
            #print(f"Y in range = {Y_in_range}")


            ax.scatter(
                X_in_range[:,3],
                X_in_range[:,1],
                Y_in_range[:,0],
                s=20,
                color=colors[igamma],
                edgecolors="black",
                zorder=1+igamma+1
            )

        # plot the scatter plot
        n_plot = 3000
        X_plot_mesh = np.zeros((30, 100))
        X_plot = np.zeros((n_plot, 4))
        ct = 0
        gamma_vec = np.linspace(0.0, 4.0, 30)
        AR_vec = np.log(np.linspace(0.1, 10.0, 100))
        for igamma in range(30):
            for iAR in range(100):
                X_plot[ct, :] = np.array(
                    [avg_xi, AR_vec[iAR], avg_zeta, gamma_vec[igamma]]
                )
                ct += 1

        Kplot = np.array(
            [
                [
                    kernel(X_train[i, :], X_plot[j, :], theta0)
                    for i in range(n_train)
                ]
                for j in range(n_plot)
            ]
        )
        f_plot = Kplot @ alpha

        # make meshgrid of outputs
        GAMMA = np.zeros((30, 100))
        AR = np.zeros((30, 100))
        KMIN = np.zeros((30, 100))
        ct = 0
        for igamma in range(30):
            for iAR in range(100):
                GAMMA[igamma, iAR] = gamma_vec[igamma]
                AR[igamma, iAR] = AR_vec[iAR]
                KMIN[igamma, iAR] = f_plot[ct]
                ct += 1

        # plot the model curve
        # Creating plot
        face_colors = cm.jet((KMIN - 0.8) / np.log(10.0))
        ax.plot_surface(
            GAMMA,
            AR,
            KMIN,
            antialiased=False,
            facecolors=face_colors,
            alpha=0.4,
            zorder=1,
        )

        # save the figure
        ax.set_xlabel(r"$\log(1+\gamma)$")
        ax.set_ylabel(r"$log(\rho_0)$")
        if args.load == "Nx":
            ax.set_zlabel(r"$log(N_{11,cr}^*)$")
        else:
            ax.set_zlabel(r"$log(N_{12,cr}^*)$")
        ax.set_ylim3d(np.log(0.1), np.log(10.0))
        #ax.set_zlim3d(0.0, np.log(50.0))
        #ax.set_zlim3d(1.0, 3.0)
        ax.view_init(elev=20, azim=20, roll=0)
        plt.gca().invert_xaxis()
        # plt.title(f"")
        plt.show()
        # plt.savefig(os.path.join(GP_folder, f"gamma-3d.png"), dpi=400)
        plt.close(f"3d rho_0, gamma, lam_star")

