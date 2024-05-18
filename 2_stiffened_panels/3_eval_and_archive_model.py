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

np.random.seed(1234567)

# parse the arguments
parent_parser = argparse.ArgumentParser(add_help=False)
parent_parser.add_argument("--load", type=str)
parent_parser.add_argument("--plotraw", type=bool, default=False)
parent_parser.add_argument("--plotmodel", type=bool, default=False)

args = parent_parser.parse_args()

assert args.load in ["Nx", "Nxy"]

load = args.load

# load the Nxcrit dataset
load_prefix = "Nx_stiffened" if load == "Nx" else "Nxy"
csv_filename = f"{load}_stiffened"
df = pd.read_csv("data/" + csv_filename + ".csv")

# extract only the model columns
X = df[["log(xi)", "log(rho_0)", "log(1+10^3*zeta)", "log(1+gamma)"]].to_numpy()
Y = df["log(lam_star)"].to_numpy()
Y = np.reshape(Y, newshape=(Y.shape[0], 1))

N_data = X.shape[0]

# n_train = int(0.9 * N_data)
n_train = 1000 # 1000, 4000
n_test = min([4000, N_data-n_train])


def relu(x):
    return max([0.0, x])

def soft_relu(x, rho=10):
    return 1.0 / rho * np.log(1 + np.exp(rho * x))

sigma_n = 1e-1 #1e-1 was old value
def kernel(xp, xq, theta):
    # xp, xq are Nx1,Mx1 vectors (ln(xi), ln(rho_0), ln(1 + 10^3 * zeta), ln(1 + gamma))
    vec = xp - xq

    d1 = vec[1]  # first two entries
    d2 = vec[2]
    d3 = vec[3]

    # log(xi) direction
    kernel0 = xp[0] * xq[0] + 0.1
    # log(rho_0) direction
    kernel1 = (
        soft_relu(-xp[1]) * soft_relu(-xq[1]) + 0.1 +
        1.0 #* np.exp(-0.5 * (xp[3] + xq[3]) / 1.0) # 0.2
        # * np.exp(-0.5 * (d1 ** 2 / (0.2 ** 2))
        * np.exp(-0.5 * (d1 ** 2 / (0.2 + (xp[3] + xq[3]) / 2.0) ** 2))
        * soft_relu(1 - abs(xp[1]))
        * soft_relu(1 - abs(xq[1]))
    )
    # log(zeta) direction
    kernel2 = xp[2] * xq[2] + 0.5 * np.exp(-0.5 * d2 ** 2 / 9.0)
    # log(gamma) direction
    kernel3 = xp[3] * xq[3] + 0.5 * np.exp(-0.5 * d3 ** 2 / 9.0)
    return kernel0 * kernel1 * kernel2 * kernel3

print(f"Monte Carlo #data training {n_train} / {X.shape[0]} data points")

# print bounds of the data
xi = X[:,0]
print(f"\txi or x0: min {np.min(xi)}, max {np.max(xi)}")
rho0 = X[:,1]
print(f"\trho0 or x1: min {np.min(rho0)}, max {np.max(rho0)}")
zeta = X[:,2]
print(f"\tzeta or x2: min {np.min(zeta)}, max {np.max(zeta)}")
gamma = X[:,3]
print(f"\tgamma or x3: min {np.min(gamma)}, max {np.max(gamma)}")

# bins for the data (in log space)
xi_bins = [[-1.2, -0.8], [-0.8, -0.4], [-0.4, 0.0], [0.0, 0.4]]
rho0_bins = [[-2.5, -1.0], [-1.0, 0.0], [0.0, 1.0], [1.0, 2.5]]
zeta_bins = [[0.0, 1.0], [1.0, 2.0], [3.0, 4.0], [4.0, 5.0]]
gamma_bins = [[0.0, 0.1], [0.1, 1.0], [1.0, 3.0], [3.0, 5.0] ]

# randomly permute the arrays
rand_perm = np.random.permutation(N_data)
X = X[rand_perm, :]
Y = Y[rand_perm, :]

# REMOVE THE OUTLIERS in local 4d regions
# Step 1 - remove any gamma > 0 points below the curve (there aren't many)
# but these are definitely outliers / bad points
_remove_outliers = True
if _remove_outliers:
    _remove_indices = []
    _full_indices = np.array([_ for _ in range(N_data)])

    for ixi, xi_bin in enumerate(xi_bins):
        xi_mask = np.logical_and(xi_bin[0] <= X[:,0], X[:,0] <= xi_bin[1])
        avg_xi = 0.5 * (xi_bin[0] + xi_bin[1])

        for izeta, zeta_bin in enumerate(zeta_bins):
            zeta_mask = np.logical_and(zeta_bin[0] <= X[:,2], X[:,2] <= zeta_bin[1])
            avg_zeta = 0.5 * (zeta_bin[0] + zeta_bin[1])
            xi_zeta_mask = np.logical_and(xi_mask, zeta_mask)

            for irho0, rho0_bin in enumerate(rho0_bins):
                rho0_mask = np.logical_and(rho0_bin[0] <= X[:,1], X[:,1] <= rho0_bin[1])
                mask = np.logical_and(xi_zeta_mask, rho0_mask)

                # for each gamma > 0 data point
                gm_lg0_mask = np.logical_and(mask, X[:,3] > 0.0)
                gm_eq0_mask = np.logical_and(mask, X[:,3] == 0.0)

                if np.sum(gm_eq0_mask) == 0: continue

                # do a local linear regression of the gamma == 0.0 data
                # so that we can check whether the data point is below gamma == 0.0 surface
                X_local = X[gm_eq0_mask, :3]
                Y_local = Y[gm_eq0_mask, :]
                wstar_local = np.linalg.solve(X_local.T @ X_local + 1e-4, X_local.T @ Y_local)

                # now for each point in the gamma > 0 set check if below the gamma == 0.0 surface
                X_gm_lg0 = X[gm_lg0_mask, :3]
                Y_gm_lg0 = Y[gm_lg0_mask, :]
                Y_pred = X_gm_lg0 @ wstar_local
                Y_resid = Y_gm_lg0 - Y_pred
                gm_lg0_indices = _full_indices[gm_lg0_mask]
                for i,glob_index in enumerate(gm_lg0_indices):
                    if Y_resid[i] < 0.0:
                        _remove_indices += [glob_index]

    n_removed = len(_remove_indices)
    print(f"removed {n_removed} outliers : now {N_data} data points left")

    # also remove some of the bad data based on inspecting the 2d plots
    # some seemed like they are not mesh converged for high AR, high gamma, or this is mode distortion?
    large_xi_mask = X[:,0] >= -0.3
    large_rho0_mask = X[:,1] >= 0.0
    large_gm_mask = X[:,3] > 0.0
    mask = np.logical_and(large_xi_mask, large_rho0_mask)
    mask = np.logical_and(mask, large_gm_mask)
    new_remove_indices = _full_indices[mask]
    _remove_indices += list(new_remove_indices)

    # keep the non outlier data
    _keep_indices = [_ for  _ in range(N_data) if not(_ in _remove_indices)]
    X = X[_keep_indices,:]
    Y = Y[_keep_indices,:]

    N_data = X.shape[0]

    # n_train = int(0.9 * N_data)
    n_train = 1000 # 1000, 4000
    n_test = min([4000, N_data-n_train])

    n_removed2 = len(_remove_indices)
    print(f"removed {n_removed2-n_removed} poor mesh convergeddata points : now {N_data} data points left")
    # exit()


# 2d plots
_plot_gamma = True
_plot_zeta = True
_plot_xi = True
# 3d plots
_plot_3d = True

# make a folder for the model fitting
plots_folder = os.path.join(os.getcwd(), "plots")
sub_plots_folder = os.path.join(plots_folder, csv_filename)
GP_folder = os.path.join(sub_plots_folder, "GP")
for ifolder, folder in enumerate(
    [
        #plots_folder,
        #sub_plots_folder,
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
assert n_test > 100

# reorder the data
indices = [_ for _ in range(n_total)]
train_indices = np.random.choice(indices, size=n_train)
test_indices = [_ for _ in range(n_total) if not(_ in train_indices)]

X_train = X[train_indices, :]
X_test = X[test_indices[:n_test], :]
Y_train = Y[train_indices, :]
Y_test = Y[test_indices[:n_test], :]

# TRAIN THE MODEL HYPERPARAMETERS
# by maximizing the log marginal likelihood log p(y|X)
# or minimizing the negative log marginal likelihood
# ----------------------------------------------------
# update the local hyperparameter variables
# initial hyperparameter vector
# sigma_n, sigma_f, L1, L2, L3
# theta0 = np.array([1e-1, 3e-1, -1, 0.2, 1.0, 1.0, 0.5, 2, 0.8, 1.0, 0.2])
# theta0 = [9.99999999e-05, 9.99999997e-05, 1.59340141e-01, 7.26056398e-01,
#  9.27506384e-01, 4.98424731e-01, 7.75092936e-01, 9.96708608e-01,
#  9.99999998e-05, 9.99999998e-05, 6.96879851e-01, 1.00000000e-04,]
# theta0 = [9.99999999e-05, 1.00000000e-01, 1.70095486e-01, 8.42616423e-01,
#          9.78301139e-01, 4.05578917e-01, 7.89718868e-01, 9.15860119e-01,
#          1.00000000e-01, 6.92185488e-02, 8.37836523e-01, 1.00547797e-04,]
# theta = [1.00000000e-04 1.30342193e-01 2.10713893e-01 9.99999999e-05
#  8.50836331e-01 3.36062544e-01 8.12693340e-01 5.48897610e-01
#  1.00000000e-01 1.00000000e-02 1.00000000e-01 1.00000000e-04]

# v1
# theta0 = np.array([1e-1, 3e-1, 0.2, 1.0, 1.0, 0.5, 0.8, 1.0, 0.2, 0.2, 1.0, 1e-2])
# v2 - increase gamma length
# theta0 = np.array([1e-1, 3e-1, 0.2, 1.0, 1.0, 2.0, 0.8, 1.0, 0.2, 0.1, 1.0, 1e-2, 0.0])
# v3 - increase gamma an dzeta length again
# theta0 = np.array([1e-1, 3e-1, 0.2, 1.0, 1.0, 3.0, 3.0, 1.0, 0.2, 0.1, 1.0, 1e-2, 0.0])
# v4 - make zeta direction mostly linear
# theta0 = np.array([1e-1, 3e-1, 0.2, 1.0, 1.0, 3.0, 3.0, 0.2, 1.0, 0.1, 1.0, 1e-2, 0.0])
# v5 - make S4 smaller and add modified L1 (larger L1_rho distance at higher gamma values => so smoother there)
theta0 = np.array([1e-1, 3e-1, 0.2, 0.1, 1.0, 3.0, 3.0, 0.2, 1.0, 0.1, 1.0, 1e-2, 1.0])

# y is the training set observations
y = Y_train


# plot the model and some of the data near the model range in D*=1, AR from 0.5 to 5.0, b/h=100
# ---------------------------------------------------------------------------------------------

if args.plotraw:
    print(f"start plot")
    # get the available colors
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

    plt.style.use(niceplots.get_style())

    if _plot_gamma:
        
        print(f"start 2d gamma plots...")

        for ixi, xi_bin in enumerate(xi_bins):
            xi_mask = np.logical_and(xi_bin[0] <= X[:,0], X[:,0] <= xi_bin[1])
            avg_xi = 0.5 * (xi_bin[0] + xi_bin[1])

            for izeta, zeta_bin in enumerate(zeta_bins):
                zeta_mask = np.logical_and(zeta_bin[0] <= X[:,2], X[:,2] <= zeta_bin[1])
                avg_zeta = 0.5 * (zeta_bin[0] + zeta_bin[1])
                xi_zeta_mask = np.logical_and(xi_mask, zeta_mask)

                if np.sum(xi_zeta_mask) == 0: continue

                plt.figure(f"xi = {avg_xi:.2f}, zeta = {avg_zeta:.2f}", figsize=(8,6))

                colors = plt.cm.jet(np.linspace(0.0, 1.0, len(gamma_bins)))

                for igamma,gamma_bin in enumerate(gamma_bins[::-1]):

                    gamma_mask = np.logical_and(gamma_bin[0] <= X[:,3], X[:,3] <= gamma_bin[1])
                    mask = np.logical_and(xi_zeta_mask, gamma_mask)

                    if np.sum(mask) == 0: continue

                    X_in_range = X[mask,:]
                    Y_in_range = Y[mask,:]

                    plt.plot(
                        X_in_range[:,1],
                        Y_in_range[:,0],
                        "o",
                        color=colors[igamma],
                        zorder=1+igamma,
                        label=f"gamma in [{gamma_bin[0]:.0f},{gamma_bin[1]:.0f}]"
                    )

                plt.legend()
                plt.xlabel(r"$\log{\rho_0}$")
                plt.ylabel(r"$N_{cr}^*$")

                plt.savefig(os.path.join(GP_folder, f"2d-gamma_xi{ixi}_zeta{izeta}.png"), dpi=400)
                plt.close(f"xi = {avg_xi:.2f}, zeta = {avg_zeta:.2f}")

    if _plot_zeta:
        
        print(f"start 2d zeta plots...")

        for ixi, xi_bin in enumerate(xi_bins):
            xi_mask = np.logical_and(xi_bin[0] <= X[:,0], X[:,0] <= xi_bin[1])
            avg_xi = 0.5 * (xi_bin[0] + xi_bin[1])

            for igamma,gamma_bin in enumerate(gamma_bins[::-1]):
                gamma_mask = np.logical_and(gamma_bin[0] <= X[:,3], X[:,3] <= gamma_bin[1])
                avg_gamma = 0.5 * (gamma_bin[0] + gamma_bin[1])
                xi_gamma_mask = np.logical_and(xi_mask, gamma_mask)

                for izeta, zeta_bin in enumerate(zeta_bins):
                    zeta_mask = np.logical_and(zeta_bin[0] <= X[:,2], X[:,2] <= zeta_bin[1])
                    avg_zeta = 0.5 * (zeta_bin[0] + zeta_bin[1])
                    mask = np.logical_and(xi_gamma_mask, zeta_mask)

                    if np.sum(mask) == 0: continue

                    plt.figure(f"xi = {avg_xi:.2f}, gamma = {avg_gamma:.2f}", figsize=(8,6))

                    colors = plt.cm.jet(np.linspace(0.0, 1.0, len(zeta_bins)))

                

                    if np.sum(mask) == 0: continue

                    X_in_range = X[mask,:]
                    Y_in_range = Y[mask,:]

                    plt.plot(
                        X_in_range[:,1],
                        Y_in_range[:,0],
                        "o",
                        color=colors[izeta],
                        zorder=1+izeta,
                        label=f"Lzeta in [{zeta_bin[0]:.0f},{zeta_bin[1]:.0f}]"
                    )

                plt.legend()
                plt.xlabel(r"$\log{\rho_0}$")
                plt.ylabel(r"$N_{cr}^*$")

                plt.savefig(os.path.join(GP_folder, f"2d-zeta_xi{ixi}_gamma{igamma}.png"), dpi=400)
                plt.close(f"xi = {avg_xi:.2f}, gamma = {avg_gamma:.2f}")  

    if _plot_xi:
        
        print(f"start 2d zeta plots...")

        for igamma,gamma_bin in enumerate(gamma_bins[::-1]):
            gamma_mask = np.logical_and(gamma_bin[0] <= X[:,3], X[:,3] <= gamma_bin[1])
            avg_gamma = 0.5 * (gamma_bin[0] + gamma_bin[1])

            for izeta, zeta_bin in enumerate(zeta_bins):
                zeta_mask = np.logical_and(zeta_bin[0] <= X[:,2], X[:,2] <= zeta_bin[1])
                avg_zeta = 0.5 * (zeta_bin[0] + zeta_bin[1])
                gamma_zeta_mask = np.logical_and(gamma_mask, zeta_mask)

                for ixi, xi_bin in enumerate(xi_bins):
                    xi_mask = np.logical_and(xi_bin[0] <= X[:,0], X[:,0] <= xi_bin[1])
                    avg_xi = 0.5 * (xi_bin[0] + xi_bin[1])
                    mask = np.logical_and(gamma_zeta_mask, xi_mask)

                    if np.sum(mask) == 0: continue

                    plt.figure(f"zeta = {avg_zeta:.2f}, gamma = {avg_gamma:.2f}", figsize=(8,6))

                    colors = plt.cm.jet(np.linspace(0.0, 1.0, len(xi_bins)))

            
                    if np.sum(mask) == 0: continue

                    X_in_range = X[mask,:]
                    Y_in_range = Y[mask,:]

                    plt.plot(
                        X_in_range[:,1],
                        Y_in_range[:,0],
                        "o",
                        color=colors[ixi],
                        zorder=1+ixi,
                        label=f"Lxi in [{xi_bin[0]:.1f},{xi_bin[1]:.1f}]"
                    )

                plt.legend()
                plt.xlabel(r"$\log{\rho_0}$")
                plt.ylabel(r"$N_{cr}^*$")

                plt.savefig(os.path.join(GP_folder, f"2d-xi_zeta{izeta}_gamma{igamma}.png"), dpi=400)
                plt.close(f"zeta = {avg_zeta:.2f}, gamma = {avg_gamma:.2f}")  

# exit()

# compute the training kernel matrix
K_y = np.array(
    [
        [kernel(X_train[i, :], X_train[j, :], theta0) for i in range(n_train)]
        for j in range(n_train)
    ]
) + sigma_n ** 2 * np.eye(n_train)

#print(f"K_y = {K_y}")
#exit()

alpha = np.linalg.solve(K_y, y)

if args.plotmodel:

    n_plot_2d = 100
    rho0_vec = np.linspace(-2.5, 2.5, n_plot_2d)

    if _plot_gamma:
        
        print(f"start 2d gamma plots...")

        for ixi, xi_bin in enumerate(xi_bins):
            xi_mask = np.logical_and(xi_bin[0] <= X[:,0], X[:,0] <= xi_bin[1])
            avg_xi = 0.5 * (xi_bin[0] + xi_bin[1])

            for izeta, zeta_bin in enumerate(zeta_bins):
                zeta_mask = np.logical_and(zeta_bin[0] <= X[:,2], X[:,2] <= zeta_bin[1])
                avg_zeta = 0.5 * (zeta_bin[0] + zeta_bin[1])
                xi_zeta_mask = np.logical_and(xi_mask, zeta_mask)

                if np.sum(xi_zeta_mask) == 0: continue

                plt.figure(f"xi = {avg_xi:.2f}, zeta = {avg_zeta:.2f}", figsize=(8,6))

                colors = plt.cm.jet(np.linspace(0.0, 1.0, len(gamma_bins)))

                for igamma,gamma_bin in enumerate(gamma_bins[::-1]):

                    gamma_mask = np.logical_and(gamma_bin[0] <= X[:,3], X[:,3] <= gamma_bin[1])
                    avg_gamma = 0.5 * (gamma_bin[0] + gamma_bin[1])
                    mask = np.logical_and(xi_zeta_mask, gamma_mask)

                    if np.sum(mask) == 0: continue

                    X_in_range = X[mask,:]
                    Y_in_range = Y[mask,:]

                    plt.plot(
                        X_in_range[:,1],
                        Y_in_range[:,0],
                        "o",
                        color=colors[igamma],
                        zorder=1+igamma,
                        label=f"gamma in [{gamma_bin[0]:.0f},{gamma_bin[1]:.0f}]"
                    )

                    # predict the models, with the same colors, no labels
                    X_plot = np.zeros((n_plot_2d, 4))
                    for irho, crho0 in enumerate(rho0_vec):
                        X_plot[irho,:] = np.array([avg_xi, crho0, avg_zeta, avg_gamma])[:]

                    Kplot = np.array(
                        [
                            [
                                kernel(X_train[i, :], X_plot[j, :], theta0)
                                for i in range(n_train)
                            ]
                            for j in range(n_plot_2d)
                        ]
                    )
                    f_plot = Kplot @ alpha

                    plt.plot(
                        rho0_vec,
                        f_plot,
                        "--",
                        color=colors[igamma],
                        zorder=
                    )

                plt.legend()
                plt.xlabel(r"$\log{\rho_0}$")
                plt.ylabel(r"$N_{cr}^*$")

                plt.savefig(os.path.join(GP_folder, f"2d-gamma_xi{ixi}_zeta{izeta}.png"), dpi=400)
                plt.close(f"xi = {avg_xi:.2f}, zeta = {avg_zeta:.2f}")

    if _plot_zeta:
        
        print(f"start 2d zeta plots...")

        for ixi, xi_bin in enumerate(xi_bins):
            xi_mask = np.logical_and(xi_bin[0] <= X[:,0], X[:,0] <= xi_bin[1])
            avg_xi = 0.5 * (xi_bin[0] + xi_bin[1])

            for igamma,gamma_bin in enumerate(gamma_bins[::-1]):
                gamma_mask = np.logical_and(gamma_bin[0] <= X[:,3], X[:,3] <= gamma_bin[1])
                avg_gamma = 0.5 * (gamma_bin[0] + gamma_bin[1])
                xi_gamma_mask = np.logical_and(xi_mask, gamma_mask)

                for izeta, zeta_bin in enumerate(zeta_bins):
                    zeta_mask = np.logical_and(zeta_bin[0] <= X[:,2], X[:,2] <= zeta_bin[1])
                    avg_zeta = 0.5 * (zeta_bin[0] + zeta_bin[1])
                    mask = np.logical_and(xi_gamma_mask, zeta_mask)

                    if np.sum(mask) == 0: continue

                    plt.figure(f"xi = {avg_xi:.2f}, gamma = {avg_gamma:.2f}", figsize=(8,6))

                    colors = plt.cm.jet(np.linspace(0.0, 1.0, len(zeta_bins)))

                

                    if np.sum(mask) == 0: continue

                    X_in_range = X[mask,:]
                    Y_in_range = Y[mask,:]

                    plt.plot(
                        X_in_range[:,1],
                        Y_in_range[:,0],
                        "o",
                        color=colors[izeta],
                        zorder=1+izeta,
                        label=f"Lzeta in [{zeta_bin[0]:.0f},{zeta_bin[1]:.0f}]"
                    )

                plt.legend()
                plt.xlabel(r"$\log{\rho_0}$")
                plt.ylabel(r"$N_{cr}^*$")

                plt.savefig(os.path.join(GP_folder, f"2d-zeta_xi{ixi}_gamma{igamma}.png"), dpi=400)
                plt.close(f"xi = {avg_xi:.2f}, gamma = {avg_gamma:.2f}")  

    if _plot_xi:
        
        print(f"start 2d zeta plots...")

        for igamma,gamma_bin in enumerate(gamma_bins[::-1]):
            gamma_mask = np.logical_and(gamma_bin[0] <= X[:,3], X[:,3] <= gamma_bin[1])
            avg_gamma = 0.5 * (gamma_bin[0] + gamma_bin[1])

            for izeta, zeta_bin in enumerate(zeta_bins):
                zeta_mask = np.logical_and(zeta_bin[0] <= X[:,2], X[:,2] <= zeta_bin[1])
                avg_zeta = 0.5 * (zeta_bin[0] + zeta_bin[1])
                gamma_zeta_mask = np.logical_and(gamma_mask, zeta_mask)

                for ixi, xi_bin in enumerate(xi_bins):
                    xi_mask = np.logical_and(xi_bin[0] <= X[:,0], X[:,0] <= xi_bin[1])
                    avg_xi = 0.5 * (xi_bin[0] + xi_bin[1])
                    mask = np.logical_and(gamma_zeta_mask, xi_mask)

                    if np.sum(mask) == 0: continue

                    plt.figure(f"zeta = {avg_zeta:.2f}, gamma = {avg_gamma:.2f}", figsize=(8,6))

                    colors = plt.cm.jet(np.linspace(0.0, 1.0, len(xi_bins)))

            
                    if np.sum(mask) == 0: continue

                    X_in_range = X[mask,:]
                    Y_in_range = Y[mask,:]

                    plt.plot(
                        X_in_range[:,1],
                        Y_in_range[:,0],
                        "o",
                        color=colors[ixi],
                        zorder=1+ixi,
                        label=f"Lxi in [{xi_bin[0]:.1f},{xi_bin[1]:.1f}]"
                    )

                    # plot the model now
                    

                plt.legend()
                plt.xlabel(r"$\log{\rho_0}$")
                plt.ylabel(r"$N_{cr}^*$")

                plt.savefig(os.path.join(GP_folder, f"2d-xi_zeta{izeta}_gamma{igamma}.png"), dpi=400)
                plt.close(f"zeta = {avg_zeta:.2f}, gamma = {avg_gamma:.2f}")  

    if _plot_3d:

        # 3d plot of rho_0, gamma, lam_star for a particular xi and zeta range
        # xi_bin = [-0.5, 0.5]
        # smaller values of xi have higher gamma
        xi_bin = [-1.15, -0.85]
        # xi_bin = [-2.0, 2.0]
        xi_mask = np.logical_and(xi_bin[0] <= X[:,0], X[:,0] <= xi_bin[1])
        avg_xi = -1.15
        # zeta_bin = [0.0, 8.0]
        zeta_bin = [0.3, 1.25]
        # zeta_bin = [0.0, 8.0]
        zeta_mask = np.logical_and(zeta_bin[0] <= X[:,2], X[:,2] <= zeta_bin[1])
        avg_zeta = 0.8
        xi_zeta_mask = np.logical_and(xi_mask, zeta_mask)

        plt.figure(f"3d rho_0, gamma, lam_star")
        ax = plt.axes(projection="3d", computed_zorder=False)

        colors = plt.cm.jet(np.linspace(0.0, 1.0, len(gamma_bins)))

        for igamma,gamma_bin in enumerate(gamma_bins):

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
                zorder=2+igamma
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

# predict and report the relative error on the test dataset
K_test_cross = np.array(
    [
        [
            kernel(X_train[i, :], X_test[j, :], theta0)
            for i in range(n_train)
        ]
        for j in range(n_test)
    ]
)
Y_test_pred = K_test_cross @ alpha

crit_loads = np.exp(Y_test)
crit_loads_pred = np.exp(Y_test_pred)

abs_err = crit_loads_pred - crit_loads
rel_err = abs(abs_err / crit_loads)
avg_rel_err = np.mean(rel_err)
if args.plotraw or args.plotmodel:
    print(f"\n\n\n")
print(f"\navg rel err from n_train={n_train} on test set of n_test={n_test} = {avg_rel_err}")

# print out which data points have the highest relative error as this might help me improve the model
neg_avg_rel_err = -1.0 * rel_err
sort_indices = np.argsort(neg_avg_rel_err[:,0])
n_worst = 100
hdl = open("axial-model-debug.txt", "w")
#(f"sort indices = {sort_indices}")
for i,sort_ind in enumerate(sort_indices[:n_worst]): # first 10 
    hdl.write(f"sort ind = {type(sort_ind)}\n")
    x_test = X_test[sort_ind,:]
    crit_load = crit_loads[sort_ind,0]
    crit_load_pred = crit_loads_pred[sort_ind,0]
    hdl.write(f"{sort_ind} - lxi={x_test[0]:.3f}, lrho0={x_test[1]:.3f}, l(1+gamma)={x_test[3]:.3f}, l(1+10^3*zeta)={x_test[2]:.3f}\n")
    xi = np.exp(x_test[0])
    rho0 = np.exp(x_test[1])
    gamma = np.exp(x_test[3]) - 1.0
    zeta = (np.exp(x_test[2]) - 1.0) / 1000.0
    c_rel_err = (crit_load_pred - crit_load) / crit_load
    hdl.write(f"\txi = {xi:.3f}, rho0 = {rho0:.3f}, gamma = {gamma:.3f}, zeta = {zeta:.3f}\n")
    hdl.write(f"\tcrit_load = {crit_load:.3f}, crit_load_pred = {crit_load_pred:.3f}\n")
    hdl.write(f"\trel err = {c_rel_err:.3e}\n")
hdl.close()


# archive the data to the format of the 
filename = "axialGP.csv" if args.load == "Nx" else "shearGP.csv"
output_csv = "../archived_models/" + filename
# [log(xi), log(rho0), log(1+gamma), log(1+10^3 * zeta)]
dataframe_dict = {
    "log(xi)" : X_train[:,0],
    "log(rho0)" : X_train[:,1],
    "log(1+gamma)" : X_train[:,3],
    "log(1+10^3*zeta)" : X_train[:,2], # gamma,zeta are flipped to the order used in TACS
    "alpha" : alpha[:,0],
}
model_df = pd.DataFrame(dataframe_dict)
model_df.to_csv(output_csv)