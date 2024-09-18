import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import niceplots, scipy, time, os
import argparse
from mpl_toolkits import mplot3d
from matplotlib import cm
import shutil
from mpi4py import MPI
import ml_buckling as mlb

comm = MPI.COMM_WORLD

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
df = pd.read_csv("_raw_data/" + csv_filename + ".csv")

new_filename = "_data/" + csv_filename + ".csv"

# extract only the model columns
# TODO : if need more inputs => could maybe try adding log(E11/E22) in as a parameter?
# or also log(E11/G12)
X = df[["Dstar", "a0/b0", "b/h"]].to_numpy()
Y = df["kmin"].to_numpy()
Y = np.reshape(Y, newshape=(Y.shape[0], 1))

print(f"Monte Carlo #data = {X.shape[0]}")
N_data = X.shape[0]

n_train = int(0.9 * N_data)


# for each data point in the raw data add a new parameter
# zeta = A66/A11 * (b/h)^2
materials = df["material"].to_numpy()
zeta = np.zeros((materials.shape[0],))
ply_angles = df["ply_angle"].to_numpy()

for i in range(materials.shape[0]):
    material_name = materials[i]
    ply_angle = ply_angles[i]
    h = 1.0
    b = (
        h * X[i, 2]
    )  # verified that different b values don't influence non-dim buckling load
    AR = 1.0
    a = b * AR
    material = mlb.UnstiffenedPlateAnalysis.get_material_from_str(material_name)
    new_plate: mlb.UnstiffenedPlateAnalysis = material(
        comm,
        bdf_file="plate.bdf",
        a=a,
        b=b,
        h=h,
        ply_angle=ply_angle,
    )

    zeta[i] = new_plate.zeta


# REMOVE THE OUTLIERS in local 4d regions
# loop over different slenderness bins
slender_bins = [
    [10.0, 20.0],
    [20.0, 50.0],
    [50.0, 100.0],
    [100.0, 200.0],
]  # [5.0, 10.0],
Dstar_bins = [[0.25 * i, 0.25 * (i + 1)] for i in range(7)]
# added smaller and larger bins here cause we miss some of the outliers near the higher a0/b0 with less data
aff_AR_bins = (
    [[0.5 * i, 0.5 * (i + 1)] for i in range(4)]
    + [[1.0 * i, 1.0 * (i + 1)] for i in range(2, 5)]
    + [[2.0, 10.0]]
)

_plot_outliers = False

# make a folder for the model fitting
plots_folder = os.path.join(os.getcwd(), "plots")
sub_plots_folder = os.path.join(plots_folder, csv_filename)
wo_outliers_folder = os.path.join(sub_plots_folder, "model-no-outliers")
w_outliers_folder = os.path.join(sub_plots_folder, "model-w-outliers")
for ifolder, folder in enumerate(
    [
        plots_folder,
        sub_plots_folder,
        wo_outliers_folder,
        w_outliers_folder,
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
# print(f"n data = {n_data}")
global_outlier_mask = np.full((n_data,), False, dtype=bool)

plt_ct = 0

for ibin, bin in enumerate(slender_bins):
    if ibin < len(slender_bins) - 1:
        mask1 = np.logical_and(bin[0] <= slenderness, slenderness < bin[1])
    else:
        mask1 = np.logical_and(bin[0] <= slenderness, slenderness <= bin[1])
    if np.sum(mask1) == 0:
        continue

    for iDstar, Dstar_bin in enumerate(Dstar_bins):
        if iDstar < len(Dstar_bins) - 1:
            mask2 = np.logical_and(Dstar_bin[0] <= Dstar, Dstar < Dstar_bin[1])
        else:
            mask2 = np.logical_and(Dstar_bin[0] <= Dstar, Dstar <= Dstar_bin[1])

        # limit to local regions in D*, slenderness
        layer2_mask = np.logical_and(mask1, mask2)

        for iAR, AR_bin in enumerate(aff_AR_bins):
            if iAR < len(aff_AR_bins) - 1:
                mask3 = np.logical_and(AR_bin[0] <= affine_AR, affine_AR < AR_bin[1])
            else:
                mask3 = np.logical_and(AR_bin[0] <= affine_AR, affine_AR <= AR_bin[1])

            mask = np.logical_and(layer2_mask, mask3)
            # and also now in affine_AR
            N = np.sum(mask)

            # print(f"aspect ratio bin = {AR_bin}, N = {N}")

            if N < 5:
                continue

            local_indices = np.where(mask)[0]

            # compute a local polynomial fit - mean and covariance
            # eliminate local outliers
            X_local = X[mask, :]
            t_local = Y[mask, :]

            loc_AR = X_local[:, 1:2]
            X_fit = np.concatenate([np.ones((N, 1)), loc_AR, loc_AR ** 2], axis=1)
            # print(f"X_fit shape = {X_fit.shape}")
            # print(f"t_local shape = {t_local.shape}")
            w_hat = np.linalg.solve(X_fit.T @ X_fit, X_fit.T @ t_local)
            # print(f"w_hat shape = {w_hat.shape}")

            # compute the local noise
            t_pred = X_fit @ w_hat
            resids = t_local - t_pred
            variance = float(1 / N * resids.T @ resids)
            sigma = np.sqrt(variance)

            # compute plotted model
            if _plot_outliers:
                plot_AR = np.linspace(AR_bin[0], AR_bin[1], 20).reshape((20, 1))
                X_plot = np.concatenate(
                    [np.ones((20, 1)), plot_AR, plot_AR ** 2], axis=1
                )
                t_plot = X_plot @ w_hat

                # plot the local polynomial model and its variance range
                plt.figure("temp")
                ax = plt.subplot(111)
                plt.margins(x=0.05, y=0.05)
                plt.title(
                    f"b/h - [{bin[0]},{bin[1]}], D* - [{Dstar_bin[0]},{Dstar_bin[1]}]"
                )
                plt.plot(loc_AR, t_local, "ko")
                plt.plot(plot_AR, t_plot - 2 * sigma, "r--", label=r"$\mu-2 \sigma$")
                plt.plot(plot_AR, t_plot, "b-", label=r"$\mu$")
                plt.plot(plot_AR, t_plot + 2 * sigma, "b--", label=r"$\mu+2 \sigma$")
                plt.legend()
                plt.xlabel(r"$a_0/b_0$")
                plt.ylabel(r"$k_{x_0}$")
                plt.savefig(
                    os.path.join(
                        wo_outliers_folder, f"slender{ibin}_Dstar{iDstar}_AR{iAR}.png"
                    ),
                    dpi=400,
                )
                plt_ct += 1
                plt.close("temp")

            # get all of the datapoints that lie outside the +- 2 sigma bounds
            z_scores = abs(resids[:, 0]) / sigma
            # print(f"zscores = {z_scores}")
            outlier_mask = z_scores >= 2.0
            if np.sum(outlier_mask) == 0:
                continue  # no outliers found so exit
            X_outliers = X_local[outlier_mask, :]

            # now show plot with outliers indicated for sanity check
            if _plot_outliers:
                loc_AR_noout = loc_AR[np.logical_not(outlier_mask), :]
                t_local_noout = t_local[np.logical_not(outlier_mask), :]
                loc_AR_out = loc_AR[outlier_mask, :]
                t_local_out = t_local[outlier_mask, :]
                plt.figure("temp", figsize=(8, 6))
                ax = plt.subplot(111)
                plt.margins(x=0.05, y=0.05)
                plt.title(
                    f"b/h - [{bin[0]},{bin[1]}], D* - [{Dstar_bin[0]},{Dstar_bin[1]}]"
                )
                ax.fill_between(
                    plot_AR[:, 0],
                    t_plot[:, 0] - 2 * sigma,
                    t_plot[:, 0] + 2 * sigma,
                    color="g",
                )
                ax.plot(plot_AR, t_plot + 2 * sigma, "b--", label=r"$\mu+2 \sigma$")
                ax.plot(plot_AR, t_plot, "b-", label=r"$\mu$")
                ax.plot(plot_AR, t_plot - 2 * sigma, "r--", label=r"$\mu-2 \sigma$")
                ax.plot(loc_AR_noout, t_local_noout, "ko", label="data")
                ax.plot(loc_AR_out, t_local_out, "ro", label="outlier")
                plt.legend()
                plt.xlabel(r"$a_0/b_0$")
                plt.ylabel(r"$k_{x_0}$")
                box = ax.get_position()
                ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                # Put a legend to the right of the current axis
                ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
                plt.savefig(
                    os.path.join(
                        w_outliers_folder, f"slender{ibin}_Dstar{iDstar}_AR{iAR}.png"
                    ),
                    dpi=400,
                )
                plt_ct += 1
                plt.close("temp")

            # now record the outliers to the global outlier indices and remove them
            n_outliers = np.sum(outlier_mask)
            # print(f"local indices = {local_indices[:3]} type {type(local_indices)}")
            # print(f"outlier mask = {outlier_mask} type {type(outlier_mask)} shape = {outlier_mask.shape}")
            _outlier_indices = local_indices[outlier_mask]
            # print(f"local outlier indices = {_outlier_indices} shape = {_outlier_indices.shape}")
            for _outlier in _outlier_indices:
                global_outlier_mask[_outlier] = True

# print(f"global outlier mask = {global_outlier_mask}")
print(f"num outliers = {np.sum(global_outlier_mask)}")
# exit()

# replace X[0] xi with log(1+xi)
X[:, 0] += 1.0

# replace X[2] the b/h slenderness column with zeta
X[:, 2] = 1.0 + 1000.0 * zeta[:]

# remove the outliers from the dataset
keep_mask = np.logical_not(global_outlier_mask)
X = X[keep_mask, :]
Y = Y[keep_mask, :]

# double the shear raw data since it was wrong eps_12 vs gamma_12
if args.load == "Nxy":
    Y[:, :] *= 2.0

# convert xi, a0/b0, b/h and kmin to log space
X[:, :] = np.log(X[:, :])
Y[:, 0:] = np.log(Y[:, 0:])

data_dict = {"x0": X[:, 0], "x1": X[:, 1], "x2": X[:, 2], "y": Y[:, 0]}

# write out to csv file in _data folder
new_df = pd.DataFrame(data_dict)
new_df.to_csv(new_filename)
