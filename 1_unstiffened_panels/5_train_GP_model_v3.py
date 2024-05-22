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
parent_parser.add_argument("--BC", type=str)

args = parent_parser.parse_args()

assert args.load in ["Nx", "Nxy", "axial", "shear"]
assert args.BC in ["SS", "CL"]

print(f"args.load = {args.load}")
if args.load in ["Nx", "axial"]:
    load = "Nx"
else:
    load = "Nxy"
BC = args.BC

# load the Nxcrit dataset
load_prefix = "Nxcrit" if load == "Nx" else "Nxycrit"
csv_filename = f"{load_prefix}_{BC}"
print(f"csv filename = {csv_filename}")
df = pd.read_csv("_data/" + csv_filename + ".csv")

# extract only the model columns
# TODO : if need more inputs => could maybe try adding log(E11/E22) in as a parameter?
# or also log(E11/G12)
X = df[["x0", "x1", "x2"]].to_numpy()
Y = df["y"].to_numpy()
Y = np.reshape(Y, newshape=(Y.shape[0], 1))

print(f"Monte Carlo #data = {X.shape[0]}")
N_data = X.shape[0]

n_train = int(0.9 * N_data)
# n_train = 100

# REMOVE THE OUTLIERS in local 4d regions
# loop over different slenderness bins
slender_bins = [
    [0.0, 0.6],
    [0.6, 1.0],
    [1.0, 2.0],
    [2.0, 3.0],
    [3.0, 4.5],
]
Dstar_bins = [[0.25 * i, 0.25 * (i + 1)] for i in range(7)]
# added smaller and larger bins here cause we miss some of the outliers near the higher a0/b0 with less data
aff_AR_bins = (
    [[0.5 * i, 0.5 * (i + 1)] for i in range(4)]
    + [[1.0 * i, 1.0 * (i + 1)] for i in range(2, 5)]
    + [[2.0, 10.0]]
)

_plot_outliers = False
_plot = True
_plot_Dstar_2d = False
_plot_slender_2d = False
_plot_3d = True
_plot_model_fit = False
_plot_model_fit_xi = False

# make a folder for the model fitting
plots_folder = os.path.join(os.getcwd(), "plots")
sub_plots_folder = os.path.join(plots_folder, csv_filename)
wo_outliers_folder = os.path.join(sub_plots_folder, "model-no-outliers")
w_outliers_folder = os.path.join(sub_plots_folder, "model-w-outliers")
GP_folder = os.path.join(sub_plots_folder, "GP_v2")
for ifolder, folder in enumerate(
    [
        plots_folder,
        sub_plots_folder,
        wo_outliers_folder,
        w_outliers_folder,
        GP_folder,
    ]
):
    if ifolder > 0 and os.path.exists(folder):
        shutil.rmtree(folder)
    if ifolder in [2, 3]:
        _mk_folder = _plot_outliers
    else:
        _mk_folder = True
    if not os.path.exists(folder) and _mk_folder:
        os.mkdir(folder)

plt.style.use(niceplots.get_style())
Dstar = X[:, 0]
affine_AR = X[:, 1]
slenderness = X[:, 2]
kx0 = Y[:, 0]

n_data = Dstar.shape[0]

# split into training and test datasets
n_total = X.shape[0]
n_test = n_total - n_train
assert n_test > 100

X_train = X[:n_train, :]
X_test = X[n_train:, :]
Y_train = Y[:n_train, :]
Y_test = Y[n_train:, :]

# print(f"Y train = {Y_train.shape}")

# TRAIN THE MODEL HYPERPARAMETERS
# by maximizing the log marginal likelihood log p(y|X)
# or minimizing the negative log marginal likelihood
# ----------------------------------------------------

# y is the training set observations
y = Y_train

# update the local hyperparameter variables
# initial hyperparameter vector
# sigma_n, sigma_f, L1, L2, L3
theta0 = np.array([1e-1, 3e-1, -1, 0.2, 1.0, 1.0, 0.3, 2])
sigma_n = 1e-2


def relu(x):
    return max([0.0, x])


def soft_relu(x, rho=10):
    return 1.0 / rho * np.log(1 + np.exp(rho * x))


def kernel(xp, xq, theta):
    # xp, xq are Nx1,Mx1 vectors (D*, a0/b0, ln(b/h))
    vec = xp - xq

    S1 = theta[0]
    S2 = theta[1]
    c = theta[2]
    L1 = theta[3]
    S4 = theta[4]
    S5 = theta[5]
    L2 = theta[6]
    alpha_1 = theta[7]

    d1 = vec[1]  # first two entries
    d2 = vec[2]

    # kernel in
    kernel0 = S1 ** 2 + S2 ** 2 * soft_relu(xp[0] - c, alpha_1) * soft_relu(
        xq[0] - c, alpha_1
    )
    kernel1 = (
        np.exp(-0.5 * (d1 ** 2 / L1 ** 2))
        * soft_relu(1 - abs(xp[1]))
        * soft_relu(1 - abs(xq[1]))
        + S4
        + S5 * soft_relu(-xp[1]) * soft_relu(-xq[1])
    )
    kernel2 = np.exp(-0.5 * d2 ** 2 / L2 ** 2)
    return kernel0 * kernel1 * kernel2


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

    if _plot_model_fit_xi:  #
        # iterate over the different slender,D* bins
        for ibin, bin in enumerate(slender_bins):
            slender_bin = [np.log(bin[0]), np.log(bin[1])]
            avg_log_slender = 0.5 * (slender_bin[0] + slender_bin[1])
            mask1 = np.logical_and(slender_bin[0] <= X[:, 2], X[:, 2] <= slender_bin[1])
            if np.sum(mask1) == 0:
                continue

            plt.figure("check model", figsize=(8, 6))
            plt.margins(x=0.05, y=0.05)
            plt.title(f"b/h in [{bin[0]},{bin[1]}]")
            ax = plt.subplot(111)

            # pick high aspect ratio for xi fit
            AR_fit = np.log(3.0) <= X[:, 1]
            mask = np.logical_and(mask1, AR_fit)
            if np.sum(mask) == 0:
                continue

            # predict the GP curve
            n_plot = 100
            X_plot = np.zeros((n_plot, 3))
            X_plot[:, 0] = np.log(np.linspace(0.4, 1.5, n_plot))
            X_plot[:, 1] = np.log(4.0)  # np.log(np.linspace(0.1, 10.0, n_plot))
            X_plot[:, 2] = avg_log_slender

            Kplot_train = np.array(
                [
                    [
                        kernel(X_train[i, :], X_plot[j, :], theta0)
                        for i in range(n_train)
                    ]
                    for j in range(n_plot)
                ]
            )
            f_plot = Kplot_train @ alpha

            Kpp = np.array(
                [
                    [kernel(X_plot[i, :], X_plot[j, :], theta0) for i in range(n_plot)]
                    for j in range(n_plot)
                ]
            )
            Kpp += sigma_n ** 2 * np.eye(n_plot)

            # cholesky decomp to find the covariance and standard deviations of the model
            L = np.linalg.cholesky(K_y)  # decomposes K_train into L * L^T

            # now also get the covariance of the plot dataset
            # solve (L L^T) A = K_cross^T by first solving L A1 = Kcross^T
            A1 = scipy.linalg.solve_triangular(L, Kplot_train.T, lower=True)
            # solve L^T A = A1
            A = scipy.linalg.solve_triangular(L.T, A1, lower=False)

            cov_Y_plot = Kpp - Kplot_train @ A  # + sigma_n**2 * np.eye(n_plot)
            # cov_Y_plot = Kpp - Kplot_train @ A + sigma_n**2 * np.eye(n_plot)
            var_Y_plot = np.diag(cov_Y_plot).reshape((n_plot, 1))
            std_dev = np.sqrt(var_Y_plot)

            print(f"f_plot = {f_plot[:10,0]}")
            print(f"var plot = {var_Y_plot[:10,0]}")
            print(f"std_dev = {std_dev[:10,0]}", flush=True)
            # exit()

            # plot data in certain range of the training set
            X_in_range = X[mask, :]
            Y_in_range = Y[mask, :]

            log_xi_in_range = X_in_range[:, 0]

            # plot the raw data and the model in this range
            # plot ax fill between
            ax.fill_between(
                x=X_plot[:, 0],
                y1=f_plot[:, 0] - 3 * std_dev[:, 0],
                y2=f_plot[:, 0] + 3 * std_dev[:, 0],
                label="3-sigma",
            )
            ax.fill_between(
                x=X_plot[:, 0],
                y1=f_plot[:, 0] - std_dev[:, 0],
                y2=f_plot[:, 0] + std_dev[:, 0],
                label="1-sigma",
            )
            ax.plot(
                X_plot[:, 0], f_plot[:, 0], "k", label="mean"
            )  # , label=f"D*-[{Dstar_bin[0]},{Dstar_bin[1]}]""
            ax.plot(
                log_xi_in_range, Y_in_range, "o", markersize=4, label="train-data"
            )  # , label=f"D*-[{Dstar_bin[0]},{Dstar_bin[1]}]""

            model_dict = {
                "xi": X_plot[:, 0],
                "mean": f_plot[:, 0],
                "std_dev": std_dev[:, 0],
            }
            model_df = pd.DataFrame(model_dict)
            model_df_filename = os.path.join(GP_folder, f"model-fit-xi_model.csv")
            print(f"writing model df filename {model_df_filename}")
            model_df.to_csv(model_df_filename)
            data_dict = {
                "AR": log_xi_in_range,
                "lam": Y_in_range[:, 0],
            }
            data_df = pd.DataFrame(data_dict)
            data_df_filename = os.path.join(GP_folder, f"model-fit-xi_data.csv")
            print(f"writing data df filename {data_df_filename}")
            data_df.to_csv(data_df_filename)

            # outside of for loop save the plot
            plt.xlabel(r"$log(\xi)$")
            plt.ylabel(r"$log(\lambda_{min}^*)$")
            plt.legend()
            plt.xlim(np.log(0.4), np.log(1.5))
            plt.ylim(0.0, np.log(20.0))
            # box = ax.get_position()
            # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            # # Put a legend to the right of the current axis
            # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.savefig(os.path.join(GP_folder, f"xi{ibin}-model-fit.png"), dpi=400)
            plt.close("check model")

    if _plot_model_fit:
        # iterate over the different slender,D* bins
        for ibin, bin in enumerate(slender_bins):
            slender_bin = [np.log(bin[0]), np.log(bin[1])]
            avg_log_slender = 0.5 * (slender_bin[0] + slender_bin[1])
            mask1 = np.logical_and(slender_bin[0] <= X[:, 2], X[:, 2] <= slender_bin[1])
            if np.sum(mask1) == 0:
                continue

            plt.figure("check model", figsize=(8, 6))
            plt.margins(x=0.05, y=0.05)
            plt.title(f"b/h in [{bin[0]},{bin[1]}]")
            ax = plt.subplot(111)

            for iDstar, Dstar_bin in enumerate(Dstar_bins):
                if iDstar != 3:
                    continue  # only do one of them for this plot
                log_Dstar_bin = np.log(np.array(Dstar_bin))
                mask2 = np.logical_and(
                    log_Dstar_bin[0] <= X[:, 0], X[:, 0] <= log_Dstar_bin[1]
                )
                avg_Dstar = 0.5 * (log_Dstar_bin[0] + log_Dstar_bin[1])

                mask = np.logical_and(mask1, mask2)
                if np.sum(mask) == 0:
                    continue

                # predict the GP curve
                n_plot = 300
                X_plot = np.zeros((n_plot, 3))
                X_plot[:, 0] = avg_Dstar
                X_plot[:, 1] = np.log(np.linspace(0.1, 10.0, n_plot))
                X_plot[:, 2] = avg_log_slender

                Kplot_train = np.array(
                    [
                        [
                            kernel(X_train[i, :], X_plot[j, :], theta0)
                            for i in range(n_train)
                        ]
                        for j in range(n_plot)
                    ]
                )
                f_plot = Kplot_train @ alpha

                Kpp = np.array(
                    [
                        [
                            kernel(X_plot[i, :], X_plot[j, :], theta0)
                            for i in range(n_plot)
                        ]
                        for j in range(n_plot)
                    ]
                )
                Kpp += sigma_n ** 2 * np.eye(n_plot)

                # cholesky decomp to find the covariance and standard deviations of the model
                L = np.linalg.cholesky(K_y)  # decomposes K_train into L * L^T

                # now also get the covariance of the plot dataset
                # solve (L L^T) A = K_cross^T by first solving L A1 = Kcross^T
                A1 = scipy.linalg.solve_triangular(L, Kplot_train.T, lower=True)
                # solve L^T A = A1
                A = scipy.linalg.solve_triangular(L.T, A1, lower=False)

                cov_Y_plot = Kpp - Kplot_train @ A  # + sigma_n**2 * np.eye(n_plot)
                # cov_Y_plot = Kpp - Kplot_train @ A + sigma_n**2 * np.eye(n_plot)
                var_Y_plot = np.diag(cov_Y_plot).reshape((n_plot, 1))
                std_dev = np.sqrt(var_Y_plot)

                print(f"f_plot = {f_plot[:10,0]}")
                print(f"var plot = {var_Y_plot[:10,0]}")
                print(f"std_dev = {std_dev[:10,0]}", flush=True)
                # exit()

                # plot data in certain range of the training set
                mask = np.logical_and(mask1, mask2)
                X_in_range = X[mask, :]
                Y_in_range = Y[mask, :]

                AR_in_range = X_in_range[:, 1]

                model_dict = {
                    "rho_0": X_plot[:, 1],
                    "mean": f_plot[:, 0],
                    "std_dev": std_dev[:, 0],
                }
                model_df = pd.DataFrame(model_dict)
                model_df_filename = os.path.join(
                    GP_folder, f"slender{ibin}-model-fit_model.csv"
                )
                print(f"writing model df filename {model_df_filename}")
                model_df.to_csv(model_df_filename)
                data_dict = {
                    "AR": AR_in_range,
                    "lam": Y_in_range[:, 0],
                }
                data_df = pd.DataFrame(data_dict)
                data_df_filename = os.path.join(
                    GP_folder, f"slender{ibin}-model-fit_data.csv"
                )
                print(f"writing data df filename {data_df_filename}")
                data_df.to_csv(data_df_filename)

                # plot the raw data and the model in this range
                # plot ax fill between
                ax.fill_between(
                    x=X_plot[:, 1],
                    y1=f_plot[:, 0] - 3 * std_dev[:, 0],
                    y2=f_plot[:, 0] + 3 * std_dev[:, 0],
                    label="3-sigma",
                )
                ax.fill_between(
                    x=X_plot[:, 1],
                    y1=f_plot[:, 0] - std_dev[:, 0],
                    y2=f_plot[:, 0] + std_dev[:, 0],
                    label="1-sigma",
                )
                ax.plot(
                    X_plot[:, 1], f_plot[:, 0], "k", label="mean"
                )  # , label=f"D*-[{Dstar_bin[0]},{Dstar_bin[1]}]""
                ax.plot(
                    AR_in_range, Y_in_range, "o", markersize=4, label="train-data"
                )  # , label=f"D*-[{Dstar_bin[0]},{Dstar_bin[1]}]""

            # outside of for loop save the plot
            plt.xlabel(r"$log(\rho_0)$")
            plt.ylabel(r"$log(\lambda_{min}^*)$")
            plt.legend()
            plt.xlim(np.log(0.1), np.log(20.0))
            plt.ylim(0.0, np.log(20.0))
            # box = ax.get_position()
            # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            # # Put a legend to the right of the current axis
            # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.savefig(
                os.path.join(GP_folder, f"slender{ibin}-model-fit.png"), dpi=400
            )
            plt.close("check model")

    if _plot_3d:
        # plot the 3D version of the GP curve for each slenderness value
        # iterate over the different slender,D* bins
        for ibin, bin in enumerate(slender_bins):
            # slender_bin = [np.log(bin[0]), np.log(bin[1])]
            slender_bin = bin
            avg_log_slender = 0.5 * (slender_bin[0] + slender_bin[1])
            mask1 = np.logical_and(slender_bin[0] <= X[:, 2], X[:, 2] <= slender_bin[1])
            if np.sum(mask1) == 0:
                continue

            fig = plt.figure("3d-GP", figsize=(14, 9))
            ax = plt.axes(projection="3d", computed_zorder=False)

            # plot data in certain range of the training set
            for iDstar, Dstar_bin in enumerate(Dstar_bins):
                log_Dstar_bin = np.log(1.0 + np.array(Dstar_bin))
                mask2 = np.logical_and(
                    log_Dstar_bin[0] <= X[:, 0], X[:, 0] <= log_Dstar_bin[1]
                )
                avg_Dstar = 0.5 * (log_Dstar_bin[0] + log_Dstar_bin[1])

                mask = np.logical_and(mask1, mask2)
                if np.sum(mask) == 0:
                    continue
                X_in_range = X[mask, :]
                Y_in_range = Y[mask, :]
                ax.scatter(
                    X_in_range[:, 0],
                    X_in_range[:, 1],
                    Y_in_range[:, 0],
                    s=40,
                    edgecolors="black",
                    zorder=1 + iDstar,
                )

            n_plot = 3000
            X_plot_mesh = np.zeros((30, 100))
            X_plot = np.zeros((n_plot, 3))
            ct = 0
            Dstar_vec = np.log(1.0+np.linspace(0.25, 1.75, 30))
            AR_vec = np.log(np.linspace(0.1, 10.0, 100))
            for iDstar in range(30):
                for iAR in range(100):
                    X_plot[ct, :] = np.array(
                        [Dstar_vec[iDstar], AR_vec[iAR], avg_log_slender]
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
            DSTAR = np.zeros((30, 100))
            AR = np.zeros((30, 100))
            KMIN = np.zeros((30, 100))
            ct = 0
            for iDstar in range(30):
                for iAR in range(100):
                    DSTAR[iDstar, iAR] = Dstar_vec[iDstar]
                    AR[iDstar, iAR] = AR_vec[iAR]
                    KMIN[iDstar, iAR] = f_plot[ct]
                    ct += 1

            # plot the model curve
            # Creating plot
            face_colors = cm.jet((KMIN - 0.8) / np.log(10.0))
            ax.plot_surface(
                DSTAR,
                AR,
                KMIN,
                antialiased=False,
                facecolors=face_colors,
                alpha=0.4,
                zorder=1,
            )

            # save the figure
            ax.set_xlabel(r"$\log(1+\xi)$")
            ax.set_ylabel(r"$log(\rho_0)$")
            ax.set_zlabel(r"$log(N_{11,cr}^*)$")
            ax.set_ylim3d(np.log(0.1), np.log(10.0))
            ax.set_zlim3d(0.0, np.log(30.0))
            ax.view_init(elev=20, azim=20, roll=0)
            plt.gca().invert_xaxis()
            # plt.title(f"b/h in [{bin[0]},{bin[1]}]")
            # plt.show()
            plt.savefig(os.path.join(GP_folder, f"3d-slender-{ibin}.png"), dpi=400)
            plt.close("3d-GP")

    if _plot_slender_2d:
        # iterate over the different slender,D* bins
        for ibin, bin in enumerate(slender_bins):
            slender_bin = [np.log(bin[0]), np.log(bin[1])]
            avg_log_slender = 0.5 * (slender_bin[0] + slender_bin[1])
            mask1 = np.logical_and(slender_bin[0] <= X[:, 2], X[:, 2] <= slender_bin[1])
            if np.sum(mask1) == 0:
                continue

            plt.figure("check model", figsize=(8, 6))
            plt.margins(x=0.05, y=0.05)
            plt.title(f"b/h in [{bin[0]},{bin[1]}]")
            ax = plt.subplot(111)

            for iDstar, Dstar_bin in enumerate(Dstar_bins):
                log_Dstar_bin = np.log(np.array(Dstar_bin))
                mask2 = np.logical_and(
                    log_Dstar_bin[0] <= X[:, 0], X[:, 0] <= log_Dstar_bin[1]
                )
                avg_Dstar = 0.5 * (log_Dstar_bin[0] + log_Dstar_bin[1])

                mask = np.logical_and(mask1, mask2)
                if np.sum(mask) == 0:
                    continue

                # predict the GP curve
                n_plot = 300
                X_plot = np.zeros((n_plot, 3))
                X_plot[:, 0] = avg_Dstar
                X_plot[:, 1] = np.log(np.linspace(0.1, 10.0, n_plot))
                X_plot[:, 2] = avg_log_slender

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

                # plot data in certain range of the training set
                mask = np.logical_and(mask1, mask2)
                X_in_range = X[mask, :]
                Y_in_range = Y[mask, :]

                AR_in_range = X_in_range[:, 1]

                # plot the raw data and the model in this range
                ax.plot(
                    AR_in_range, Y_in_range, "o", color=colors[iDstar]
                )  # , label=f"D*-[{Dstar_bin[0]},{Dstar_bin[1]}]""
                ax.plot(
                    X_plot[:, 1],
                    f_plot[:, 0],
                    color=colors[iDstar],
                    label=f"xi-[{Dstar_bin[0]},{Dstar_bin[1]}]",
                )

            # outside of for loop save the plot
            plt.xlabel(r"$log(\rho_0)$")
            plt.ylabel(r"$log(\lambda_{min}^*)$")
            plt.legend()
            plt.xlim(np.log(0.1), np.log(20.0))
            plt.ylim(0.0, np.log(20.0))
            # box = ax.get_position()
            # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            # # Put a legend to the right of the current axis
            # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.savefig(os.path.join(GP_folder, f"slender{ibin}.png"), dpi=400)
            plt.close("check model")

    if _plot_Dstar_2d:
        # then do the same but reverse D* then slender for loop
        # iterate over the different slender,D* bins
        for iDstar, Dstar_bin in enumerate(Dstar_bins):
            log_Dstar_bin = np.log(np.array(Dstar_bin))
            mask2 = np.logical_and(
                log_Dstar_bin[0] <= X[:, 0], X[:, 0] <= log_Dstar_bin[1]
            )
            avg_Dstar = 0.5 * (log_Dstar_bin[0] + log_Dstar_bin[1])
            if np.sum(mask2) == 0:
                continue

            plt.figure("check model", figsize=(8, 6))
            plt.margins(x=0.05, y=0.05)
            plt.title(f"xi in [{Dstar_bin[0]},{Dstar_bin[1]}]")
            ax = plt.subplot(111)

            for ibin, bin in enumerate(slender_bins):
                slender_bin = [np.log(bin[0]), np.log(bin[1])]
                avg_log_slender = 0.5 * (slender_bin[0] + slender_bin[1])
                mask1 = np.logical_and(
                    slender_bin[0] <= X[:, 2], X[:, 2] <= slender_bin[1]
                )

                mask = np.logical_and(mask1, mask2)
                if np.sum(mask) == 0:
                    continue

                # predict the GP curve
                n_plot = 300
                X_plot = np.zeros((n_plot, 3))
                X_plot[:, 0] = avg_Dstar
                X_plot[:, 1] = np.log(np.linspace(0.1, 10.0, n_plot))
                X_plot[:, 2] = avg_log_slender

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

                # plot data in certain range of the training set
                mask = np.logical_and(mask1, mask2)
                X_in_range = X[mask, :]
                Y_in_range = Y[mask, :]

                AR_in_range = X_in_range[:, 1]

                # plot the raw data and the model in this range
                ax.plot(
                    AR_in_range, Y_in_range, "o", color=colors[ibin]
                )  # , label=f"D*-[{Dstar_bin[0]},{Dstar_bin[1]}]""
                ax.plot(
                    X_plot[:, 1],
                    f_plot[:, 0],
                    color=colors[ibin],
                    label=f"b/h-[{bin[0]},{bin[1]}]",
                )

            # outside of for loop save the plot
            plt.xlabel(r"$log(\rho_0)$")
            plt.ylabel(r"$log(\lambda_{min}^*)$")
            plt.legend()
            plt.xlim(np.log(0.1), np.log(20.0))
            plt.ylim(0.0, np.log(20.0))
            # box = ax.get_position()
            # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            # # Put a legend to the right of the current axis
            # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.savefig(os.path.join(GP_folder, f"Dstar{iDstar}.png"), dpi=400)
            plt.close("check model")

# predict against the test dataset
# --------------------------------------------------------------------------------------------------
# K(X*,X)
Kstar = np.array(
    [
        [kernel(X_train[i, :], X_test[j, :], theta0) for i in range(n_train)]
        for j in range(n_test)
    ]
)

f_test = Kstar @ alpha
# compute the RMSE on the test dataset
test_resid = Y_test - f_test
# print(f"test resid = {test_resid}")
RMSE = np.sqrt(1.0 / n_test * float(test_resid.T @ test_resid))

# norm RMSE
norm_resid = (Y_test - f_test) / Y_test
avg_rel_error = np.sum(np.abs(norm_resid)) / n_test

# write this to file in the GP folder
txt_file = os.path.join(GP_folder, "model-fit.txt")
hdl = open(txt_file, "w")
hdl.write(f"RMSE test = {RMSE}")
hdl.write(f"avg relative error = {avg_rel_error}")
hdl.close()
