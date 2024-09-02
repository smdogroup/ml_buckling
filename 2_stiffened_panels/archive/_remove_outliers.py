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
parent_parser.add_argument("--load", type=str, default="Nx")

args = parent_parser.parse_args()

assert args.load in ["Nx", "Nxy"]

load = args.load

# load the Nxcrit dataset
load_prefix = "Nx_stiffened" if load == "Nx" else "Nxy"
csv_filename = f"{load}_stiffened"
df = pd.read_csv("data/" + csv_filename + ".csv")

# extract only the model columns
X = df[["log(1+xi)", "log(rho_0)", "log(1+10^3*zeta)", "log(1+gamma)"]].to_numpy()
Y = df["log(lam_star)"].to_numpy()
Y = np.reshape(Y, newshape=(Y.shape[0], 1))

N_data = X.shape[0]

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
xi_bins = [[0.2, 0.4], [0.4, 0.6], [0.6, 0.8], [0.8, 1.0]]
rho0_bins = [[-2.5, -1.0], [-1.0, 0.0], [0.0, 1.0], [1.0, 2.5]]
zeta_bins = [[0.0, 0.1], [0.1, 0.5], [0.5, 1.0], [1.0, 2.0]]
gamma_bins = [[0.0, 0.1], [0.1, 1.0], [1.0, 3.0], [3.0, 5.0] ]

n_train = 3000 # 1000, 4000
n_test = min([4000, N_data-n_train])

# REMOVE THE OUTLIERS in local 4d regions
# Step 1 - remove any gamma > 0 points below the curve (there aren't many)
# but these are definitely outliers / bad points
_remove_outliers = args.load == "Nx" # TBD on using it for the Nxy data
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

    """
    Please note, I have removed some of the potentially bad data points here
    to make the model better. Some values of high zeta, xi, gamma are hard to mesh converge
    depending on how many stiffeners there are. Data quality is very important in training machine learning models.
    Especially if you want them to extrapolate well to high values of gamma, xi, zeta as best you can.
    Some case studies on individual FEA models is probably also warranted. Also, some trends here might be correct
    as high values of gamma for instance might reduce the gamma slope due to mode distortion. But this is unclear.
    And needs more investigation first.
    """
    # also remove some of the bad data based on inspecting the 2d plots
    # some seemed like they are not mesh converged for high AR, high gamma, or this is mode distortion?
    large_xi_mask = X[:,0] >= 0.8
    large_rho0_mask = X[:,1] >= 0.0
    large_gm_mask = X[:,3] > 0.0
    mask = np.logical_and(large_xi_mask, large_rho0_mask)
    mask = np.logical_and(mask, large_gm_mask)
    new_remove_indices = _full_indices[mask]
    _remove_indices += list(new_remove_indices)

    # remove zeta > 2 from dataset as it messes up the zeta profiles
    # zeta > 2 is very unrealistic for aircraft wings as it represents pretty thick plates
    # and wings are lightweight, thin plate structures.
    zeta_mask = X[:,2] > 2.0
    _remove_indices += list(_full_indices[zeta_mask])

    # remove xi > 1.0 as the stiffened data is too close to unstiffened data here as gamma inc
    # that it messes up the slopes (might be mesh convergence related) => hard to mesh converge high xi, gamma
    # other parts of the literature also state that 0 < xi < 1.0 for all realistic designs
    #xi_mask = X[:,0] >= 0.8
    xi_mask = X[:,0] >= 0.4
    #_remove_indices += list(_full_indices[xi_mask])

    # remove last xi_bin
    #xi_bins = xi_bins[:-1]


    # keep the non outlier data
    _keep_indices = [_ for  _ in range(N_data) if not(_ in _remove_indices)]
    X = X[_keep_indices,:]
    Y = Y[_keep_indices,:]

    N_data = X.shape[0]

    # update n_test based on remaining data
    n_test = min([4000, N_data-n_train])

    n_removed2 = len(_remove_indices)
    print(f"removed {n_removed2-n_removed} data points outside of realistic design bounds : now {N_data} data points left")
    # exit()

# remove any additional regions of bad data from observing the model and convergence study results..
# unstiffened panel data is good quality and self-consistent
gamma_zero_mask = X[:,3] < 0.01
# gamma > 0 models worked best with low xi
xi_mask = np.logical_and(0.2 <= X[:,0], X[:,0] <= 0.4)
zeta_mask = np.logical_and(0.0 <= X[:,2], X[:,2] <= 1.0)
gamma_large_mask = X[:,3] > 0.0
stiff_mask = np.logical_and(xi_mask, zeta_mask)
stiff_mask = np.logical_and(stiff_mask, gamma_large_mask)

verified_data_mask = np.logical_or(gamma_zero_mask, stiff_mask)

X = X[verified_data_mask,:]
Y = Y[verified_data_mask,:]

print(f"removed unverified models => only {X.shape[0]} models left")

# write out to Nx_stiffened2.csv
# X = df[["log(1+xi)", "log(rho_0)", "log(1+10^3*zeta)", "log(1+gamma)"]].to_numpy()
# Y = df["log(lam_star)"].to_numpy()

df_dict = {
    "log(1+xi)" : X[:,0], 
    "log(rho_0)" : X[:,1], 
    "log(1+10^3*zeta)" : X[:,2], 
    "log(1+gamma)" : X[:,3],
    "log(lam_star)" : Y[:,0],
}
df2 = pd.DataFrame(df_dict)
df2.to_csv("data/Nx_stiffened2.csv")