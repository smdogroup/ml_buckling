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

# get the dataset
# ---------------
df = pd.read_csv("Nxy_raw_stiffened.csv")

# extract only the model columns
X0 = df[["xi", "rho_0", "log10(zeta)", "gamma"]].to_numpy()
Y = df["eig_FEA"].to_numpy()
Y = np.reshape(Y, newshape=(Y.shape[0], 1))

# transform
X = X0.copy()
X[:,0] = np.log(1.0 + X0[:,0])
X[:,1] = np.log(X0[:,1])
X[:,2] = np.log(1.0 + 1e3 * np.power(10.0, X0[:,2]))
X[:,3] = np.log(1 + X0[:,3])
Y = np.log(Y)

# plot 2d gamma plots
# -------------------

xi_bins = [[0.2, 0.4], [0.4, 0.6], [0.6, 0.8], [0.8, 1.05]]
rho0_bins = [[-2.5, -1.0], [-1.0, 0.0], [0.0, 1.0], [1.0, 2.5]]
zeta_bins = [[0.0, 0.1], [0.1, 0.5], [0.5, 1.0], [1.0, 2.5]]
rho0_bin = [-2.5, 0.0]
gamma_vec = np.linspace(0.0, 2.8, 6+1)
gamma_bins = [[gamma_vec[i], gamma_vec[i+1]] for i in range(6)]

# adjust eigenvalues due to local mode decay, still hard to capture
for xi_bin in xi_bins:
    for zeta_bin in zeta_bins:
        for gamma_bin in gamma_bins:
            
            xi_mask = np.logical_and(xi_bin[0] <= X[:,0], X[:,0] <= xi_bin[1])
            zeta_mask = np.logical_and(zeta_bin[0] <= X[:,2], X[:,2] <= zeta_bin[1])
            gam_mask = np.logical_and(gamma_bin[0] <= X[:,-1], X[:,-1] <= gamma_bin[1])
            full_mask = np.logical_and(xi_mask, zeta_mask)
            full_mask = np.logical_and(full_mask, gam_mask)

            rho0_bin = [-2.5, 0.0]
            rho0_mask = np.logical_and(rho0_bin[0] <= X[:,1], X[:,1] <= rho0_bin[1])
            full_mask = np.logical_and(full_mask, rho0_mask)

            Xred = X[full_mask,:]
            # if full_mask[0]:
            #     print(f"{xi_bin=} {zeta_bin=} {gamma_bin=}")
            # print(f'{full_mask=}')

            # predicted value
            gamma = np.exp(Xred[:,-1]) - 1.0
            rho0 = np.exp(Xred[:,1])
            pred_N12 = 9.5 * (1.0 + gamma) / rho0**1.65
            pred_log_N12 = np.log(pred_N12)

            # local mode mixing still happening at low rho0, high gamma
            # frustrating to fix, so just adjusting the values by hand towards low rho0
            # with this smoothing step (has poor ML model otherwise for shear, not for axial though)
            # still have good ML model without this, but I want it to be a nearly perfect ML model, so I removed some outliers
            # and here I'm manually adjusting some of the local mode mixing decay
            alpha = 0.4 * (Xred[:,1] / -2.0) # smoothing factor
            Y[full_mask,0] = (1-alpha) * Y[full_mask,0] + alpha * pred_log_N12

# writeout new csv file
df_dict = {
    "rho_0" : X0[:,1],
    "xi" : X0[:,0],
    "gamma" : X0[:,3],
    "log10(zeta)" : X0[:,2],
    "eig_FEA" : np.exp(Y[:,0]),
}
df = pd.DataFrame(df_dict)
df.to_csv("Nxy_raw_stiffened_smoothed.csv", float_format="%.6f")