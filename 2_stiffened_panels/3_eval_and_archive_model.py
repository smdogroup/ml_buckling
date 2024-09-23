import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import niceplots, scipy, time, os
import argparse
from mpl_toolkits import mplot3d
from matplotlib import cm
import shutil, random
from _saved_kernel import kernel, axial_theta_opt, shear_theta_opt
import ml_buckling as mlb

"""
This time I'll try a Gaussian Process model to fit the axial critical load surrogate model
Inputs: D*, a0/b0, ln(b/h)
Output: k_x0
"""

np.random.seed(1234567)

# parse the arguments
parent_parser = argparse.ArgumentParser(add_help=False)
parent_parser.add_argument("--load", type=str, default="Nx")
parent_parser.add_argument("--ntrain", type=int, default=3000)
parent_parser.add_argument(
    "--plotraw", default=False, action=argparse.BooleanOptionalAction
)
parent_parser.add_argument(
    "--plotmodel2d", default=False, action=argparse.BooleanOptionalAction
)
parent_parser.add_argument(
    "--plotmodel3d", default=False, action=argparse.BooleanOptionalAction
)
parent_parser.add_argument(
    "--show", default=False, action=argparse.BooleanOptionalAction
)
parent_parser.add_argument(
    "--resid", default=False, action=argparse.BooleanOptionalAction
)
parent_parser.add_argument(
    "--clear", default=False, action=argparse.BooleanOptionalAction
)
parent_parser.add_argument(
    "--archive", default=False, action=argparse.BooleanOptionalAction
)
parent_parser.add_argument(
    "--doubleGP", default=False, action=argparse.BooleanOptionalAction
)
parent_parser.add_argument(
    "--eval", default=False, action=argparse.BooleanOptionalAction
)

args = parent_parser.parse_args()

assert args.load in ["Nx", "Nxy"]

load = args.load

# load the Nxcrit dataset
csv_filename = f"{load}_stiffened"
df = pd.read_csv("data/" + csv_filename + ".csv")

# extract only the model columns
X = df[["log(1+xi)", "log(rho_0)", "log(1+10^3*zeta)", "log(1+gamma)"]].to_numpy()
Y = df["log(eig_FEA)"].to_numpy()
Y = np.reshape(Y, newshape=(Y.shape[0], 1))

N_data = X.shape[0]
n_train = args.ntrain
n_test = N_data - n_train

print(f"Monte Carlo #data training {n_train} / {X.shape[0]} data points")

# print bounds of the data
xi = X[:, 0]
print(f"\txi or x0: min {np.min(xi)}, max {np.max(xi)}")
rho0 = X[:, 1]
print(f"\trho0 or x1: min {np.min(rho0)}, max {np.max(rho0)}")
zeta = X[:, 2]
print(f"\tzeta or x2: min {np.min(zeta)}, max {np.max(zeta)}")
gamma = X[:, 3]
print(f"\tgamma or x3: min {np.min(gamma)}, max {np.max(gamma)}")

# bins for the data (in log space)
xi_bins = [[0.2, 0.4], [0.4, 0.6], [0.6, 0.8], [0.8, 1.05]]
rho0_bins = [[-2.5, -1.0], [-1.0, 0.0], [0.0, 1.0], [1.0, 2.5]]
zeta_bins = [[0.0, 0.1], [0.1, 0.5], [0.5, 1.0], [1.0, 2.5]]
gamma_bins = [[0.0, 0.1], [0.1, 1.0], [1.0, 3.0], [3.0, 5.0]]

# randomly permute the arrays
rand_perm = np.random.permutation(N_data)
X = X[rand_perm, :]
Y = Y[rand_perm, :]

# 2d plots
_plot_2d = True
_plot_gamma = _plot_2d
_plot_zeta = _plot_2d
_plot_xi = _plot_2d
# 3d plots
_plot_3d = True
_plot_3d_gamma = _plot_3d
_plot_3d_xi = _plot_3d
_plot_3d_zeta = _plot_3d  # _plot_3d


def closed_form_resid(x, y):
    if args.load == "Nx":
        xi = np.exp(x[0]) - 1.0
        rho0 = np.exp(x[1])
        zeta = (np.exp(x[2]) - 1.0) / 1000.0
        gamma = np.exp(x[3]) - 1.0
        N11cr = 10000.0
        for m in range(1, 51):
            for n in range(1, 11):
                N11 = (
                    m ** 2 / rho0 ** 2 * (1.0 + gamma)
                    + rho0 ** 2 * n ** 4 / m ** 2
                    + 2.0 * xi * n ** 2
                )
                if N11 < N11cr:
                    N11cr = N11
        return y - np.log(N11cr)
    elif args.load == "Nxy":
        raise AssertionError("Not written shear one yet..")


# make a folder for the model fitting
plots_folder = os.path.join(os.getcwd(), "plots")
sub_plots_folder = os.path.join(plots_folder, csv_filename)
GP_folder = os.path.join(sub_plots_folder, "GP-resid" if args.resid else "GP")
for ifolder, folder in enumerate(
    [
        # plots_folder,
        sub_plots_folder,
        GP_folder,
    ]
):
    if ifolder > 0 and os.path.exists(folder) and args.clear:
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
test_indices = [_ for _ in range(n_total) if not (_ in train_indices)]

X_train = X[train_indices, :]
X_test = X[test_indices[:n_test], :]
Y_train = Y[train_indices, :]
Y_test = Y[test_indices[:n_test], :]

# train the model:
# ----------------
doubleGP = args.doubleGP and args.load == "Nx"
if not doubleGP:
    theta_opt = axial_theta_opt if args.load == "Nx" else shear_theta_opt
    ntheta = theta_opt.shape[0]
    sigma_n = theta_opt[ntheta-1]
    # compute the training kernel matrix
    K_y = np.array(
        [
            [kernel(X_train[i, :], X_train[j, :], theta_opt) for i in range(n_train)]
            for j in range(n_train)
        ]
    ) + sigma_n ** 2 * np.eye(n_train)

    #print(f"K_y = {K_y}")
    #exit()

    alpha = np.linalg.solve(K_y, Y_train)


else: # args.doubleGP and args.load == "Nx"

    # first GP
    ntheta = theta_a1.shape[0]
    sigma_n1 = theta_a1[ntheta-1]
    # compute the training kernel matrix
    K_y1 = np.array(
        [
            [kernel(X_train[i, :], X_train[j, :], theta_a1) for i in range(n_train)]
            for j in range(n_train)
        ]
    ) + sigma_n1 ** 2 * np.eye(n_train)
    alpha1 = np.linalg.solve(K_y1, Y_train)

    # get residuals from first GP
    K_cross_train1 = np.array(
        [
            [kernel(X_train[i, :], X_train[j, :], theta_a1) for i in range(n_train)]
            for j in range(n_train)
        ]
    )
    Y_pred1 = (K_cross_train1 @ alpha1)[0,0]
    Y_resid1 = Y_train - Y_pred1

    # second GP
    sigma_n2 = theta_a2[ntheta-1]
    K_y2 = np.array(
        [
            [kernel(X_train[i, :], X_train[j, :], theta_a2) for i in range(n_train)]
            for j in range(n_train)
        ]
    ) + sigma_n2 ** 2 * np.eye(n_train)
    alpha2 = np.linalg.solve(K_y2, Y_resid1)
    

# plot the raw data
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
            xi_mask = np.logical_and(xi_bin[0] <= X[:, 0], X[:, 0] <= xi_bin[1])
            avg_xi = 0.5 * (xi_bin[0] + xi_bin[1])

            for izeta, zeta_bin in enumerate(zeta_bins):
                zeta_mask = np.logical_and(
                    zeta_bin[0] <= X[:, 2], X[:, 2] <= zeta_bin[1]
                )
                avg_zeta = 0.5 * (zeta_bin[0] + zeta_bin[1])
                xi_zeta_mask = np.logical_and(xi_mask, zeta_mask)

                if np.sum(xi_zeta_mask) == 0:
                    continue

                plt.figure(f"xi = {avg_xi:.2f}, zeta = {avg_zeta:.2f}", figsize=(8, 6))

                colors = plt.cm.jet(np.linspace(0.0, 1.0, len(gamma_bins)))

                for igamma, gamma_bin in enumerate(gamma_bins[::-1]):

                    gamma_mask = np.logical_and(
                        gamma_bin[0] <= X[:, 3], X[:, 3] <= gamma_bin[1]
                    )
                    mask = np.logical_and(xi_zeta_mask, gamma_mask)

                    if np.sum(mask) == 0:
                        continue

                    X_in_range = X[mask, :]
                    Y_in_range = Y[mask, :]

                    plt.plot(
                        X_in_range[:, 1],
                        Y_in_range[:, 0],
                        "o",
                        color=colors[igamma],
                        zorder=1 + igamma,
                        label=f"gamma in [{gamma_bin[0]:.0f},{gamma_bin[1]:.0f}]",
                    )

                plt.legend()
                plt.xlabel(r"$\log{\rho_0}$")
                plt.ylabel(r"$N_{cr}^*$")

                if args.show:
                    plt.show()
                else:
                    plt.savefig(
                        os.path.join(GP_folder, f"2d-gamma_xi{ixi}_zeta{izeta}.png"),
                        dpi=400,
                    )
                plt.close(f"xi = {avg_xi:.2f}, zeta = {avg_zeta:.2f}")

    if _plot_zeta:

        print(f"start 2d zeta plots...")

        for ixi, xi_bin in enumerate(xi_bins):
            xi_mask = np.logical_and(xi_bin[0] <= X[:, 0], X[:, 0] <= xi_bin[1])
            avg_xi = 0.5 * (xi_bin[0] + xi_bin[1])

            for igamma, gamma_bin in enumerate(gamma_bins[::-1]):
                gamma_mask = np.logical_and(
                    gamma_bin[0] <= X[:, 3], X[:, 3] <= gamma_bin[1]
                )
                avg_gamma = 0.5 * (gamma_bin[0] + gamma_bin[1])
                xi_gamma_mask = np.logical_and(xi_mask, gamma_mask)

                for izeta, zeta_bin in enumerate(zeta_bins):
                    zeta_mask = np.logical_and(
                        zeta_bin[0] <= X[:, 2], X[:, 2] <= zeta_bin[1]
                    )
                    avg_zeta = 0.5 * (zeta_bin[0] + zeta_bin[1])
                    mask = np.logical_and(xi_gamma_mask, zeta_mask)

                    if np.sum(mask) == 0:
                        continue

                    plt.figure(
                        f"xi = {avg_xi:.2f}, gamma = {avg_gamma:.2f}", figsize=(8, 6)
                    )

                    colors = plt.cm.jet(np.linspace(0.0, 1.0, len(zeta_bins)))

                    if np.sum(mask) == 0:
                        continue

                    X_in_range = X[mask, :]
                    Y_in_range = Y[mask, :]

                    plt.plot(
                        X_in_range[:, 1],
                        Y_in_range[:, 0],
                        "o",
                        color=colors[izeta],
                        zorder=1 + izeta,
                        label=f"Lzeta in [{zeta_bin[0]:.0f},{zeta_bin[1]:.0f}]",
                    )

                plt.legend()
                plt.xlabel(r"$\log{\rho_0}$")
                plt.ylabel(r"$N_{cr}^*$")

                if args.show:
                    plt.show()
                else:
                    plt.savefig(
                        os.path.join(GP_folder, f"2d-zeta_xi{ixi}_gamma{igamma}.png"),
                        dpi=400,
                    )
                plt.close(f"xi = {avg_xi:.2f}, gamma = {avg_gamma:.2f}")

    if _plot_xi:

        print(f"start 2d zeta plots...")

        for igamma, gamma_bin in enumerate(gamma_bins[::-1]):
            gamma_mask = np.logical_and(
                gamma_bin[0] <= X[:, 3], X[:, 3] <= gamma_bin[1]
            )
            avg_gamma = 0.5 * (gamma_bin[0] + gamma_bin[1])

            for izeta, zeta_bin in enumerate(zeta_bins):
                zeta_mask = np.logical_and(
                    zeta_bin[0] <= X[:, 2], X[:, 2] <= zeta_bin[1]
                )
                avg_zeta = 0.5 * (zeta_bin[0] + zeta_bin[1])
                gamma_zeta_mask = np.logical_and(gamma_mask, zeta_mask)

                for ixi, xi_bin in enumerate(xi_bins):
                    xi_mask = np.logical_and(xi_bin[0] <= X[:, 0], X[:, 0] <= xi_bin[1])
                    avg_xi = 0.5 * (xi_bin[0] + xi_bin[1])
                    mask = np.logical_and(gamma_zeta_mask, xi_mask)

                    if np.sum(mask) == 0:
                        continue

                    plt.figure(
                        f"zeta = {avg_zeta:.2f}, gamma = {avg_gamma:.2f}",
                        figsize=(8, 6),
                    )

                    colors = plt.cm.jet(np.linspace(0.0, 1.0, len(xi_bins)))

                    if np.sum(mask) == 0:
                        continue

                    X_in_range = X[mask, :]
                    Y_in_range = Y[mask, :]

                    plt.plot(
                        X_in_range[:, 1],
                        Y_in_range[:, 0],
                        "o",
                        color=colors[ixi],
                        zorder=1 + ixi,
                        label=f"Lxi in [{xi_bin[0]:.1f},{xi_bin[1]:.1f}]",
                    )

                plt.legend()
                plt.xlabel(r"$\log{\rho_0}$")
                plt.ylabel(r"$N_{cr}^*$")

                if args.show:
                    plt.show()
                else:
                    plt.savefig(
                        os.path.join(GP_folder, f"2d-xi_zeta{izeta}_gamma{igamma}.png"),
                        dpi=400,
                    )
                plt.close(f"zeta = {avg_zeta:.2f}, gamma = {avg_gamma:.2f}")

# exit()


# test out a certain input
# debugging
xi = 0.9487
rho0 = 0.3121
gamma = 5.868
zeta = 0.0035
_Xtest = np.zeros((1, 4))
_Xtest[0, 0] = np.log(1.0 + xi)
_Xtest[0, 1] = np.log(rho0)
_Xtest[0, 2] = np.log(1.0 + 1000.0 * zeta)
_Xtest[0, 3] = np.log(1.0 + gamma)
# _Xtest[0,3] = np.log(1.0+1000.0*zeta)
print(f"Xtest = {_Xtest}")
# Ktest = np.array([[kernel(X_train[i,:], _Xtest[j,:], theta0, debug=True) for i in range(n_train)] for j in range(1)])
# print(f"Ktest = {Ktest}")
# for itrain in range(n_train):
#    print(f"mkernel = {Ktest[0,itrain]}")
#    print(f"alpha = {alpha[itrain]}")
# pred = Ktest @ alpha
# print(f"pred = {pred}")
# exit()

# train the model:
# ----------------
theta_opt = axial_theta_opt if args.load == "Nx" else shear_theta_opt
ntheta = theta_opt.shape[0]
sigma_n = theta_opt[ntheta - 1]
# compute the training kernel matrix
K_y = np.array(
    [
        [kernel(X_train[i, :], X_train[j, :], theta_opt) for i in range(n_train)]
        for j in range(n_train)
    ]
) + sigma_n ** 2 * np.eye(n_train)

# print(f"K_y = {K_y}")
# exit()

alpha = np.linalg.solve(K_y, Y_train)


# eval the model
# ---------------
if args.eval:
    zeta_mask = X_test[:,2] < 2.5
    X_test = X_test[zeta_mask, :]
    Y_test = Y_test[zeta_mask, :]
    n_test = X_test.shape[0]

    # predict and report the relative error on the test dataset
    if not doubleGP:
        K_test_cross = np.array(
            [
                [
                    kernel(X_train[i, :], X_test[j, :], theta_opt)
                    for i in range(n_train)
                ]
                for j in range(n_test)
            ]
        )
        Y_test_pred = K_test_cross @ alpha

    else: # doubleGP
        K_test_cross1 = np.array(
            [
                [
                    kernel(X_train[i, :], X_test[j, :], theta_a1)
                    for i in range(n_train)
                ]
                for j in range(n_test)
            ]
        )
        Y_pred1 = K_test_cross1 @ alpha1

        K_test_cross2 = np.array(
            [
                [
                    kernel(X_train[i, :], X_test[j, :], theta_a2)
                    for i in range(n_train)
                ]
                for j in range(n_test)
            ]
        )
        Y_pred2 = K_test_cross2 @ alpha2

        Y_pred = Y_pred1 + Y_pred2

    # now compare test to pred
    crit_loads = np.exp(Y_test)
    crit_loads_pred = np.exp(Y_test_pred)

    abs_err = crit_loads_pred - crit_loads
    rel_err = abs(abs_err / crit_loads)
    avg_rel_err = np.mean(rel_err)
    if args.plotraw or args.plotmodel2d or args.plotmodel3d:
        print(f"\n\n\n")
    print(f"\navg rel err from n_train={n_train} on test set of n_test={n_test} = {avg_rel_err}")

    # report the RMSE
    RMSE = np.sqrt(np.mean(np.square(Y_test - Y_test_pred)))
    print(f"RMSE = {RMSE}")

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


# plot the model againt the data
# ------------------------------

if args.plotmodel2d:

    n_plot_2d = 100
    rho0_vec = np.linspace(-2.5, 2.5, n_plot_2d)

    if _plot_gamma:

        print(f"start 2d gamma plots...")

        for ixi, xi_bin in enumerate(xi_bins):
            xi_mask = np.logical_and(xi_bin[0] <= X[:, 0], X[:, 0] <= xi_bin[1])
            avg_xi = 0.5 * (xi_bin[0] + xi_bin[1])

            for izeta, zeta_bin in enumerate(zeta_bins):
                zeta_mask = np.logical_and(
                    zeta_bin[0] <= X[:, 2], X[:, 2] <= zeta_bin[1]
                )
                avg_zeta = 0.5 * (zeta_bin[0] + zeta_bin[1])
                xi_zeta_mask = np.logical_and(xi_mask, zeta_mask)

                print(f"zeta {izeta} in {zeta_bin[0]}, {zeta_bin[1]}")

                plt.figure(f"xi = {avg_xi:.2f}, zeta = {avg_zeta:.2f}", figsize=(8, 6))

                colors = plt.cm.jet(np.linspace(0.0, 1.0, len(gamma_bins)))

                for igamma, gamma_bin in enumerate(gamma_bins[::-1]):

                    gamma_mask = np.logical_and(
                        gamma_bin[0] <= X[:, 3], X[:, 3] <= gamma_bin[1]
                    )
                    avg_gamma = 0.5 * (gamma_bin[0] + gamma_bin[1])
                    mask = np.logical_and(xi_zeta_mask, gamma_mask)

                    # if np.sum(mask) == 0: continue

                    X_in_range = X[mask, :]
                    Y_in_range = Y[mask, :]
                    if args.resid:
                        Y_resid = np.array(
                            [
                                closed_form_resid(X_in_range[i, :], Y_in_range[i, 0])
                                for i in range(X_in_range.shape[0])
                            ]
                        )
                        Y_in_range = Y_resid.reshape((X_in_range.shape[0], 1))

                    if np.sum(mask) != 0:
                        plt.plot(
                            X_in_range[:, 1],
                            Y_in_range[:, 0],
                            "o",
                            color=colors[igamma],
                            zorder=1 + igamma,
                        )

                    # predict the models, with the same colors, no labels
                    X_plot = np.zeros((n_plot_2d, 4))
                    for irho, crho0 in enumerate(rho0_vec):
                        X_plot[irho, :] = np.array(
                            [avg_xi, crho0, avg_zeta, avg_gamma]
                        )[:]

                    if not doubleGP:
                        Kplot = np.array(
                            [
                                [
                                    kernel(X_train[i, :], X_plot[j, :], theta_opt)
                                    for i in range(n_train)
                                ]
                                for j in range(n_plot_2d)
                            ]
                        )
                        f_plot = Kplot @ alpha
                    else: # doubleGP
                        K_plot1 = np.array(
                            [
                                [
                                    kernel(X_train[i, :], X_plot[j, :], theta_a1)
                                    for i in range(n_train)
                                ]
                                for j in range(n_plot_2d)
                            ]
                        )
                        f_plot1 = K_plot1 @ alpha1

                        K_plot2 = np.array(
                            [
                                [
                                    kernel(X_train[i, :], X_plot[j, :], theta_a2)
                                    for i in range(n_train)
                                ]
                                for j in range(n_plot_2d)
                            ]
                        )
                        f_plot2 = K_plot2 @ alpha2

                        f_plot = f_plot1 + f_plot2

                    if args.resid:
                        f_resid = np.array(
                            [
                                closed_form_resid(X_plot[i, :], f_plot[i])
                                for i in range(X_plot.shape[0])
                            ]
                        )
                        f_plot = f_resid.reshape((X_plot.shape[0], 1))

                    plt.plot(
                        rho0_vec,
                        f_plot,
                        "--",
                        color=colors[igamma],
                        zorder=1,
                        label=r"$\log(1+\gamma)"
                        + f"\ in\ [{gamma_bin[0]:.1f},{gamma_bin[1]:.1f}"
                        + r"]$",
                    )

                plt.legend()
                plt.xlabel(r"$\log{\rho_0}$")
                if args.load == "Nx":
                    plt.ylabel(r"$\log(N_{11,cr}^*)$")
                else:
                    plt.ylabel(r"$\log(N_{12,cr}^*)$")

                if args.show:
                    plt.show()
                else:
                    plt.savefig(
                        os.path.join(
                            GP_folder, f"2d-gamma-model_xi{ixi}_zeta{izeta}.png"
                        ),
                        dpi=400,
                    )
                plt.close(f"xi = {avg_xi:.2f}, zeta = {avg_zeta:.2f}")

    if _plot_xi:

        print(f"start 2d xi plots...")

        for igamma, gamma_bin in enumerate(gamma_bins):
            gamma_mask = np.logical_and(
                gamma_bin[0] <= X[:, 3], X[:, 3] <= gamma_bin[1]
            )
            avg_gamma = 0.5 * (gamma_bin[0] + gamma_bin[1])

            for izeta, zeta_bin in enumerate(zeta_bins):
                zeta_mask = np.logical_and(
                    zeta_bin[0] <= X[:, 2], X[:, 2] <= zeta_bin[1]
                )
                avg_zeta = 0.5 * (zeta_bin[0] + zeta_bin[1])
                gamma_zeta_mask = np.logical_and(gamma_mask, zeta_mask)

                for ixi, xi_bin in enumerate(xi_bins):
                    xi_mask = np.logical_and(xi_bin[0] <= X[:, 0], X[:, 0] <= xi_bin[1])
                    avg_xi = 0.5 * (xi_bin[0] + xi_bin[1])
                    mask = np.logical_and(gamma_zeta_mask, xi_mask)

                    # if np.sum(mask) == 0: continue

                    plt.figure(
                        f"zeta = {avg_zeta:.2f}, gamma = {avg_gamma:.2f}",
                        figsize=(8, 6),
                    )

                    colors = plt.cm.jet(np.linspace(0.0, 1.0, len(xi_bins)))

                    # if np.sum(mask) == 0: continue

                    X_in_range = X[mask, :]
                    Y_in_range = Y[mask, :]
                    if args.resid:
                        Y_resid = np.array(
                            [
                                closed_form_resid(X_in_range[i, :], Y_in_range[i, 0])
                                for i in range(X_in_range.shape[0])
                            ]
                        )
                        Y_in_range = Y_resid.reshape((X_in_range.shape[0], 1))

                    if np.sum(mask) != 0:
                        plt.plot(
                            X_in_range[:, 1],
                            Y_in_range[:, 0],
                            "o",
                            color=colors[ixi],
                            zorder=1 + ixi,
                        )

                    # plot the model now
                    # predict the models, with the same colors, no labels
                    X_plot = np.zeros((n_plot_2d, 4))
                    for irho, crho0 in enumerate(rho0_vec):
                        X_plot[irho, :] = np.array(
                            [avg_xi, crho0, avg_zeta, avg_gamma]
                        )[:]

                    if not doubleGP:
                        Kplot = np.array(
                            [
                                [
                                    kernel(X_train[i, :], X_plot[j, :], theta_opt)
                                    for i in range(n_train)
                                ]
                                for j in range(n_plot_2d)
                            ]
                        )
                        f_plot = Kplot @ alpha
                    else: # doubleGP
                        K_plot1 = np.array(
                            [
                                [
                                    kernel(X_train[i, :], X_plot[j, :], theta_a1)
                                    for i in range(n_train)
                                ]
                                for j in range(n_plot_2d)
                            ]
                        )
                        f_plot1 = K_plot1 @ alpha1

                        K_plot2 = np.array(
                            [
                                [
                                    kernel(X_train[i, :], X_plot[j, :], theta_a2)
                                    for i in range(n_train)
                                ]
                                for j in range(n_plot_2d)
                            ]
                        )
                        f_plot2 = K_plot2 @ alpha2

                        f_plot = f_plot1 + f_plot2

                    if args.resid:
                        f_resid = np.array(
                            [
                                closed_form_resid(X_plot[i, :], f_plot[i])
                                for i in range(X_plot.shape[0])
                            ]
                        )
                        f_plot = f_resid.reshape((X_plot.shape[0], 1))

                    plt.plot(
                        rho0_vec,
                        f_plot,
                        "--",
                        color=colors[ixi],
                        zorder=1,
                        label=r"$\log(1+\xi)"
                        f"\ in\ [{xi_bin[0]:.1f},{xi_bin[1]:.1f}" + r"]$",
                    )

                plt.legend()
                plt.xlabel(r"$\log{\rho_0}$")
                if args.load == "Nx":
                    plt.ylabel(r"$\log(N_{11,cr}^*)$")
                else:
                    plt.ylabel(r"$\log(N_{12,cr}^*)$")

                if args.show:
                    plt.show()
                else:
                    plt.savefig(
                        os.path.join(
                            GP_folder, f"2d-xi-model_zeta{izeta}_gamma{igamma}.png"
                        ),
                        dpi=400,
                    )
                plt.close(f"zeta = {avg_zeta:.2f}, gamma = {avg_gamma:.2f}")

    if _plot_zeta:

        print(f"start 2d zeta plots...")

        for ixi, xi_bin in enumerate(xi_bins):
            xi_mask = np.logical_and(xi_bin[0] <= X[:, 0], X[:, 0] <= xi_bin[1])
            avg_xi = 0.5 * (xi_bin[0] + xi_bin[1])

            for igamma, gamma_bin in enumerate(gamma_bins):
                gamma_mask = np.logical_and(
                    gamma_bin[0] <= X[:, 3], X[:, 3] <= gamma_bin[1]
                )
                avg_gamma = 0.5 * (gamma_bin[0] + gamma_bin[1])
                xi_gamma_mask = np.logical_and(xi_mask, gamma_mask)

                for izeta, zeta_bin in enumerate(zeta_bins):
                    zeta_mask = np.logical_and(
                        zeta_bin[0] <= X[:, 2], X[:, 2] <= zeta_bin[1]
                    )
                    avg_zeta = 0.5 * (zeta_bin[0] + zeta_bin[1])
                    mask = np.logical_and(xi_gamma_mask, zeta_mask)

                    # if np.sum(mask) == 0: continue

                    plt.figure(
                        f"xi = {avg_xi:.2f}, gamma = {avg_gamma:.2f}", figsize=(8, 6)
                    )

                    colors = plt.cm.jet(np.linspace(0.0, 1.0, len(zeta_bins)))

                    # if np.sum(mask) == 0: continue

                    X_in_range = X[mask, :]
                    Y_in_range = Y[mask, :]
                    if args.resid:
                        Y_resid = np.array(
                            [
                                closed_form_resid(X_in_range[i, :], Y_in_range[i, 0])
                                for i in range(X_in_range.shape[0])
                            ]
                        )
                        Y_in_range = Y_resid.reshape((X_in_range.shape[0], 1))

                    if np.sum(mask) != 0:
                        plt.plot(
                            X_in_range[:, 1],
                            Y_in_range[:, 0],
                            "o",
                            color=colors[izeta],
                            zorder=1 + izeta,
                        )

                    # predict the models, with the same colors, no labels
                    X_plot = np.zeros((n_plot_2d, 4))
                    for irho, crho0 in enumerate(rho0_vec):
                        X_plot[irho, :] = np.array(
                            [avg_xi, crho0, avg_zeta, avg_gamma]
                        )[:]

                    if not doubleGP:
                        Kplot = np.array(
                            [
                                [
                                    kernel(X_train[i, :], X_plot[j, :], theta_opt)
                                    for i in range(n_train)
                                ]
                                for j in range(n_plot_2d)
                            ]
                        )
                        f_plot = Kplot @ alpha
                    else: # doubleGP
                        K_plot1 = np.array(
                            [
                                [
                                    kernel(X_train[i, :], X_plot[j, :], theta_a1)
                                    for i in range(n_train)
                                ]
                                for j in range(n_plot_2d)
                            ]
                        )
                        f_plot1 = K_plot1 @ alpha1

                        K_plot2 = np.array(
                            [
                                [
                                    kernel(X_train[i, :], X_plot[j, :], theta_a2)
                                    for i in range(n_train)
                                ]
                                for j in range(n_plot_2d)
                            ]
                        )
                        f_plot2 = K_plot2 @ alpha2

                        f_plot = f_plot1 + f_plot2

                    if args.resid:
                        f_resid = np.array(
                            [
                                closed_form_resid(X_plot[i, :], f_plot[i])
                                for i in range(X_plot.shape[0])
                            ]
                        )
                        f_plot = f_resid.reshape((X_plot.shape[0], 1))

                    plt.plot(
                        rho0_vec,
                        f_plot,
                        "--",
                        color=colors[izeta],
                        zorder=1,
                        label=r"$\log(1+10^3\zeta)"
                        + f"\ in\ [{zeta_bin[0]:.1f},{zeta_bin[1]:.1f}"
                        + r"]$",
                    )

                plt.legend()
                plt.xlabel(r"$\log{\rho_0}$")
                if args.load == "Nx":
                    plt.ylabel(r"$\log(N_{11,cr}^*)$")
                else:
                    plt.ylabel(r"$\log(N_{12,cr}^*)$")

                if args.show:
                    plt.show()
                else:
                    plt.savefig(
                        os.path.join(
                            GP_folder, f"2d-zeta-model_xi{ixi}_gamma{igamma}.png"
                        ),
                        dpi=400,
                    )
                plt.close(f"xi = {avg_xi:.2f}, gamma = {avg_gamma:.2f}")


if args.plotmodel3d:
    if _plot_3d_gamma:

        # 3d plot of rho_0, gamma, lam_star for a particular xi and zeta range
        xi_bin = [0.5, 0.7]
        xi_mask = np.logical_and(xi_bin[0] <= X[:, 0], X[:, 0] <= xi_bin[1])
        avg_xi = 0.5 * (xi_bin[0] + xi_bin[1])

        zeta_bin = [0.2, 0.4]
        zeta_mask = np.logical_and(zeta_bin[0] <= X[:, 2], X[:, 2] <= zeta_bin[1])
        avg_zeta = 0.5 * (zeta_bin[0] + zeta_bin[1])
        xi_zeta_mask = np.logical_and(xi_mask, zeta_mask)

        plt.figure(f"3d rho_0, gamma, lam_star")
        ax = plt.axes(projection="3d", computed_zorder=False)

        colors = plt.cm.jet(np.linspace(0.0, 1.0, len(gamma_bins)))

        for igamma, gamma_bin in enumerate(gamma_bins):

            gamma_mask = np.logical_and(
                gamma_bin[0] <= X[:, 3], X[:, 3] <= gamma_bin[1]
            )
            mask = np.logical_and(xi_zeta_mask, gamma_mask)

            rho0_bin = [np.log(0.3), np.log(10.0)]
            rho0_mask = np.logical_and(
                rho0_bin[0] <= X[:,1], X[:,1] <= rho0_bin[1]
            )
            mask = np.logical_and(mask, rho0_mask)

            X_in_range = X[mask, :]
            Y_in_range = Y[mask, :]
            if args.resid:
                Y_resid = np.array(
                    [
                        closed_form_resid(X_in_range[i, :], Y_in_range[i, 0])
                        for i in range(X_in_range.shape[0])
                    ]
                )
                Y_in_range = Y_resid.reshape((X_in_range.shape[0], 1))

            # print(f"X in range = {X_in_range}")
            # print(f"Y in range = {Y_in_range}")

            ax.scatter(
                X_in_range[:, 3],
                X_in_range[:, 1],
                Y_in_range[:, 0],
                s=20,
                color=colors[igamma],
                edgecolors="black",
                zorder=2 + igamma,
            )

        # plot the scatter plot
        n_plot = 3000
        X_plot_mesh = np.zeros((30, 100))
        X_plot = np.zeros((n_plot, 4))
        ct = 0
        gamma_vec = np.linspace(0.0, 3.0, 30)
        AR_vec = np.log(np.linspace(0.3, 10.0, 100))
        for igamma in range(30):
            for iAR in range(100):
                X_plot[ct, :] = np.array(
                    [avg_xi, AR_vec[iAR], avg_zeta, gamma_vec[igamma]]
                )
                ct += 1

        # single vs doubleGP section
        if not doubleGP:
            Kplot = np.array(
                [
                    [
                        kernel(X_train[i, :], X_plot[j, :], theta_opt)
                        for i in range(n_train)
                    ]
                    for j in range(n_plot)
                ]
            )
            f_plot = Kplot @ alpha
        else: # doubleGP
            K_plot1 = np.array(
                [
                    [
                        kernel(X_train[i, :], X_plot[j, :], theta_a1)
                        for i in range(n_train)
                    ]
                    for j in range(n_plot)
                ]
            )
            f_plot1 = K_plot1 @ alpha1

            K_plot2 = np.array(
                [
                    [
                        kernel(X_train[i, :], X_plot[j, :], theta_a2)
                        for i in range(n_train)
                    ]
                    for j in range(n_plot)
                ]
            )
            f_plot2 = K_plot2 @ alpha2

            f_plot = f_plot1 + f_plot2

        if args.resid:
            f_resid = np.array(
                [
                    closed_form_resid(X_plot[i, :], f_plot[i])
                    for i in range(X_plot.shape[0])
                ]
            )
            f_plot = f_resid.reshape((X_plot.shape[0], 1))

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
        ax.set_xlabel(r"$ln(1+\gamma)$")
        ax.set_ylabel(r"$ln(\rho_0)$")
        if args.load == "Nx":
            ax.set_zlabel(r"$ln(N_{11,cr}^*)$")
        else:
            ax.set_zlabel(r"$ln(N_{12,cr}^*)$")
        ax.set_ylim3d(np.log(0.3), np.log(10.0))
        # ax.set_zlim3d(0.0, np.log(50.0))
        # ax.set_zlim3d(1.0, 3.0)
        ax.view_init(elev=20, azim=20, roll=0)
        plt.gca().invert_xaxis()
        # plt.title(f"")
        # plt.show()
        if args.show:
            plt.show()
        else:
            plt.savefig(os.path.join(GP_folder, f"gamma-3d.png"), dpi=400)
        plt.close(f"3d rho_0, gamma, lam_star")

    if _plot_3d_xi:

        # 3d plot of rho_0, gamma, lam_star for a particular xi and zeta range
        gamma_bin = [0.0, 0.1]
        gamma_mask = np.logical_and(gamma_bin[0] <= X[:, 3], X[:, 3] <= gamma_bin[1])
        avg_gamma = 0.5 * (gamma_bin[0] + gamma_bin[1])

        zeta_bin = [0.0, 1.0]
        zeta_mask = np.logical_and(zeta_bin[0] <= X[:, 2], X[:, 2] <= zeta_bin[1])
        avg_zeta = 0.5 * (zeta_bin[0] + zeta_bin[1])
        gamma_zeta_mask = np.logical_and(gamma_mask, zeta_mask)

        plt.figure(f"3d rho_0, xi, lam_star")
        ax = plt.axes(projection="3d", computed_zorder=False)

        colors = plt.cm.jet(np.linspace(0.0, 1.0, len(xi_bins)))

        for ixi, xi_bin in enumerate(xi_bins):

            xi_mask = np.logical_and(xi_bin[0] <= X[:, 0], X[:, 0] <= xi_bin[1])
            avg_xi = 0.5 * (xi_bin[0] + xi_bin[1])
            mask = np.logical_and(gamma_zeta_mask, xi_mask)

            rho0_bin = [np.log(0.3), np.log(10.0)]
            rho0_mask = np.logical_and(
                rho0_bin[0] <= X[:,1], X[:,1] <= rho0_bin[1]
            )
            mask = np.logical_and(mask, rho0_mask)

            X_in_range = X[mask, :]
            Y_in_range = Y[mask, :]
            if args.resid:
                Y_resid = np.array(
                    [
                        closed_form_resid(X_in_range[i, :], Y_in_range[i, 0])
                        for i in range(X_in_range.shape[0])
                    ]
                )
                Y_in_range = Y_resid.reshape((X_in_range.shape[0], 1))

            # print(f"X in range = {X_in_range}")
            # print(f"Y in range = {Y_in_range}")

            ax.scatter(
                X_in_range[:, 0],
                X_in_range[:, 1],
                Y_in_range[:, 0],
                s=20,
                color=colors[ixi],
                edgecolors="black",
                zorder=2 + ixi,
            )

        # plot the scatter plot
        n_plot = 3000
        X_plot_mesh = np.zeros((30, 100))
        X_plot = np.zeros((n_plot, 4))
        ct = 0
        xi_vec = np.linspace(0.2, 1.0, 30)
        AR_vec = np.log(np.linspace(0.3, 10.0, 100))
        for ixi in range(30):
            for iAR in range(100):
                X_plot[ct, :] = np.array(
                    [xi_vec[ixi], AR_vec[iAR], avg_zeta, avg_gamma]
                )
                ct += 1

        # single vs doubleGP section
        if not doubleGP:
            Kplot = np.array(
                [
                    [
                        kernel(X_train[i, :], X_plot[j, :], theta_opt)
                        for i in range(n_train)
                    ]
                    for j in range(n_plot)
                ]
            )
            f_plot = Kplot @ alpha
        else: # doubleGP
            K_plot1 = np.array(
                [
                    [
                        kernel(X_train[i, :], X_plot[j, :], theta_a1)
                        for i in range(n_train)
                    ]
                    for j in range(n_plot)
                ]
            )
            f_plot1 = K_plot1 @ alpha1

            K_plot2 = np.array(
                [
                    [
                        kernel(X_train[i, :], X_plot[j, :], theta_a2)
                        for i in range(n_train)
                    ]
                    for j in range(n_plot)
                ]
            )
            f_plot2 = K_plot2 @ alpha2

            f_plot = f_plot1 + f_plot2


        if args.resid:
            f_resid = np.array(
                [
                    closed_form_resid(X_plot[i, :], f_plot[i])
                    for i in range(X_plot.shape[0])
                ]
            )
            f_plot = f_resid.reshape((X_plot.shape[0], 1))

        # make meshgrid of outputs
        XI = np.zeros((30, 100))
        AR = np.zeros((30, 100))
        KMIN = np.zeros((30, 100))
        ct = 0
        for ixi in range(30):
            for iAR in range(100):
                XI[ixi, iAR] = xi_vec[ixi]
                AR[ixi, iAR] = AR_vec[iAR]
                KMIN[ixi, iAR] = f_plot[ct]
                ct += 1

        # plot the model curve
        # Creating plot
        face_colors = cm.jet((KMIN - 0.8) / np.log(10.0))
        ax.plot_surface(
            XI,
            AR,
            KMIN,
            antialiased=False,
            facecolors=face_colors,
            alpha=0.4,
            zorder=1,
        )

        # save the figure
        ax.set_xlabel(r"$ln(1+\xi)$")
        ax.set_ylabel(r"$ln(\rho_0)$")
        if args.load == "Nx":
            ax.set_zlabel(r"$ln(N_{11,cr}^*)$")
        else:
            ax.set_zlabel(r"$ln(N_{12,cr}^*)$")
        ax.set_ylim3d(np.log(0.3), np.log(10.0))
        # ax.set_zlim3d(0.0, np.log(50.0))
        # ax.set_zlim3d(1.0, 3.0)
        ax.view_init(elev=20, azim=20, roll=0)
        plt.gca().invert_xaxis()
        # plt.title(f"")
        # plt.show()
        if args.show:
            plt.show()
        else:
            plt.savefig(os.path.join(GP_folder, f"xi-3d.png"), dpi=400)
        plt.close(f"3d rho_0, xi, lam_star")

    if _plot_3d_zeta:

        # 3d plot of rho_0, gamma, lam_star for a particular xi and zeta range
        gamma_bin = [0.0, 0.1]
        gamma_mask = np.logical_and(gamma_bin[0] <= X[:, 3], X[:, 3] <= gamma_bin[1])
        avg_gamma = 0.5 * (gamma_bin[0] + gamma_bin[1])

        xi_bin = [0.2, 0.4]
        xi_mask = np.logical_and(xi_bin[0] <= X[:, 0], X[:, 0] <= xi_bin[1])
        avg_xi = 0.5 * (xi_bin[0] + xi_bin[1])

        gamma_xi_mask = np.logical_and(gamma_mask, xi_mask)

        plt.figure(f"3d rho_0, zeta, lam_star")
        ax = plt.axes(projection="3d", computed_zorder=False)

        colors = plt.cm.jet(np.linspace(0.0, 1.0, len(zeta_bins)))

        for izeta, zeta_bin in enumerate(zeta_bins):

            zeta_mask = np.logical_and(zeta_bin[0] <= X[:, 2], X[:, 2] <= zeta_bin[1])
            avg_zeta = 0.5 * (zeta_bin[0] + zeta_bin[1])

            mask = np.logical_and(gamma_xi_mask, zeta_mask)

            X_in_range = X[mask, :]
            Y_in_range = Y[mask, :]
            if args.resid:
                Y_resid = np.array(
                    [
                        closed_form_resid(X_in_range[i, :], Y_in_range[i, 0])
                        for i in range(X_in_range.shape[0])
                    ]
                )
                Y_in_range = Y_resid.reshape((X_in_range.shape[0], 1))

            # print(f"X in range = {X_in_range}")
            # print(f"Y in range = {Y_in_range}")

            ax.scatter(
                X_in_range[:, 2],
                X_in_range[:, 1],
                Y_in_range[:, 0],
                s=20,
                color=colors[izeta],
                edgecolors="black",
                zorder=2 + izeta,
            )

        # plot the scatter plot
        n_plot = 3000
        X_plot_mesh = np.zeros((30, 100))
        X_plot = np.zeros((n_plot, 4))
        ct = 0
        zeta_vec = np.linspace(0.0, 2.0, 30)
        AR_vec = np.log(np.linspace(0.1, 10.0, 100))
        for izeta in range(30):
            for iAR in range(100):
                X_plot[ct, :] = np.array(
                    [avg_xi, AR_vec[iAR], zeta_vec[izeta], avg_gamma]
                )
                ct += 1

        # single vs doubleGP section
        if not doubleGP:
            Kplot = np.array(
                [
                    [
                        kernel(X_train[i, :], X_plot[j, :], theta_opt)
                        for i in range(n_train)
                    ]
                    for j in range(n_plot)
                ]
            )
            f_plot = Kplot @ alpha
        else: # doubleGP
            K_plot1 = np.array(
                [
                    [
                        kernel(X_train[i, :], X_plot[j, :], theta_a1)
                        for i in range(n_train)
                    ]
                    for j in range(n_plot)
                ]
            )
            f_plot1 = K_plot1 @ alpha1

            K_plot2 = np.array(
                [
                    [
                        kernel(X_train[i, :], X_plot[j, :], theta_a2)
                        for i in range(n_train)
                    ]
                    for j in range(n_plot)
                ]
            )
            f_plot2 = K_plot2 @ alpha2

            f_plot = f_plot1 + f_plot2


        if args.resid:
            f_resid = np.array(
                [
                    closed_form_resid(X_plot[i, :], f_plot[i])
                    for i in range(X_plot.shape[0])
                ]
            )
            f_plot = f_resid.reshape((X_plot.shape[0], 1))

        # make meshgrid of outputs
        ZETA = np.zeros((30, 100))
        AR = np.zeros((30, 100))
        KMIN = np.zeros((30, 100))
        ct = 0
        for izeta in range(30):
            for iAR in range(100):
                ZETA[izeta, iAR] = zeta_vec[izeta]
                AR[izeta, iAR] = AR_vec[iAR]
                KMIN[izeta, iAR] = f_plot[ct]
                ct += 1

        # plot the model curve
        # Creating plot
        face_colors = cm.jet((KMIN - 0.8) / np.log(10.0))
        ax.plot_surface(
            ZETA,
            AR,
            KMIN,
            antialiased=False,
            facecolors=face_colors,
            alpha=0.4,
            zorder=1,
        )

        # save the figure
        ax.set_xlabel(r"$\log(1+10^3 \zeta)$")
        ax.set_ylabel(r"$log(\rho_0)$")
        if args.load == "Nx":
            ax.set_zlabel(r"$log(N_{11,cr}^*)$")
        else:
            ax.set_zlabel(r"$log(N_{12,cr}^*)$")
        ax.set_ylim3d(np.log(0.1), np.log(10.0))
        # ax.set_zlim3d(0.0, np.log(50.0))
        # ax.set_zlim3d(1.0, 3.0)
        ax.view_init(elev=20, azim=20, roll=0)
        plt.gca().invert_xaxis()
        # plt.title(f"")
        # plt.show()
        if args.show:
            plt.show()
        else:
            plt.savefig(os.path.join(GP_folder, f"zeta-3d.png"), dpi=400)
        plt.close(f"3d rho_0, zeta, lam_star")

# only eval relative error on test set for zeta < 1
# because based on the model plots it appears that the patterns break down some for that
zeta_mask = X_test[:, 2] < 2.5
X_test = X_test[zeta_mask, :]
Y_test = Y_test[zeta_mask, :]
n_test = X_test.shape[0]

# predict and report the relative error on the test dataset
K_test_cross = np.array(
    [
        [kernel(X_train[i, :], X_test[j, :], theta_opt) for i in range(n_train)]
        for j in range(n_test)
    ]
)
Y_test_pred = K_test_cross @ alpha

crit_loads = np.exp(Y_test)
crit_loads_pred = np.exp(Y_test_pred)

abs_err = crit_loads_pred - crit_loads
rel_err = abs(abs_err / crit_loads)
avg_rel_err = np.mean(rel_err)
if args.plotraw or args.plotmodel2d or args.plotmodel3d:
    print(f"\n\n\n")
print(
    f"\navg rel err from n_train={n_train} on test set of n_test={n_test} = {avg_rel_err}"
)

# report the RMSE
RMSE = np.sqrt(np.mean(np.square(Y_test - Y_test_pred)))
print(f"RMSE = {RMSE}")

# print out which data points have the highest relative error as this might help me improve the model
neg_avg_rel_err = -1.0 * rel_err
sort_indices = np.argsort(neg_avg_rel_err[:, 0])
n_worst = 100
hdl = open("axial-model-debug.txt", "w")
# (f"sort indices = {sort_indices}")
for i, sort_ind in enumerate(sort_indices[:n_worst]):  # first 10
    hdl.write(f"sort ind = {type(sort_ind)}\n")
    x_test = X_test[sort_ind, :]
    crit_load = crit_loads[sort_ind, 0]
    crit_load_pred = crit_loads_pred[sort_ind, 0]
    hdl.write(
        f"{sort_ind} - lxi={x_test[0]:.3f}, lrho0={x_test[1]:.3f}, l(1+gamma)={x_test[3]:.3f}, l(1+10^3*zeta)={x_test[2]:.3f}\n"
    )
    xi = np.exp(x_test[0])
    rho0 = np.exp(x_test[1])
    gamma = np.exp(x_test[3]) - 1.0
    zeta = (np.exp(x_test[2]) - 1.0) / 1000.0
    c_rel_err = (crit_load_pred - crit_load) / crit_load
    hdl.write(
        f"\txi = {xi:.3f}, rho0 = {rho0:.3f}, gamma = {gamma:.3f}, zeta = {zeta:.3f}\n"
    )
    hdl.write(f"\tcrit_load = {crit_load:.3f}, crit_load_pred = {crit_load_pred:.3f}\n")
    hdl.write(f"\trel err = {c_rel_err:.3e}\n")
hdl.close()


if args.archive:
    # archive the data to the format of the
    filename = "axialGP.csv" if args.load == "Nx" else "shearGP.csv"
    output_csv = "../archived_models/" + filename

    # remove the previous csv file if it exits
    # assume on serial here
    if os.path.exists(output_csv):
        os.remove(output_csv)

    # print(f"{X_train=}")

    # [log(1+xi), log(rho0), log(1+gamma), log(1+10^3 * zeta)]
    dataframe_dict = {
        "log(1+xi)": X_train[:, 0],
        "log(rho0)": X_train[:, 1],
        "log(1+gamma)": X_train[:, 3],
        "log(1+10^3*zeta)": X_train[
            :, 2
        ],  # gamma,zeta are flipped to the order used in TACS
        "alpha": alpha[:, 0],
    }
    model_df = pd.DataFrame(dataframe_dict)
    model_df.to_csv(output_csv)

    # also deploy the current theta_opt
    theta_csv = mlb.axial_theta_csv if args.load == "Nx" else mlb.shear_theta_csv
    if os.path.exists(theta_csv):
        os.remove(theta_csv)

    theta_df_dict = {
        "theta": theta_opt,
    }
    theta_df = pd.DataFrame(theta_df_dict)
    theta_df.to_csv(theta_csv)
