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
parent_parser.add_argument("--resid", type=bool, default=False)
parent_parser.add_argument("--kernel", type=int, default=9)
parent_parser.add_argument("--archive", type=bool, default=False)

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

# n_train = int(0.9 * N_data)
n_train = 3000 # 1000, 4000
n_test = min([4000, N_data-n_train])


def relu(x):
    return max([0.0, x])

def soft_relu(x, rho=10):
    return 1.0 / rho * np.log(1 + np.exp(rho * x))

def soft_abs(x, rho=10):
    return 1.0 / rho * np.log(np.exp(rho*x) + np.exp(-rho*x))

sigma_n = 1e-1 #1e-1 was old value

# this one was a pretty good model except for high gamma, middle rho0 for one region of the design
kernel_option = args.kernel # best is 7 right now
if kernel_option == 1:
    def kernel(xp, xq, theta):
        # xp, xq are Nx1,Mx1 vectors (ln(xi), ln(rho_0), ln(1 + 10^3 * zeta), ln(1 + gamma))
        vec = xp - xq

        d1 = vec[1]  # first two entries
        d2 = vec[2]
        d3 = vec[3]

        # log(xi) direction
        kernel0 = 1.0 + xp[0] * xq[0]

        #  kernel2 = xp[2] * xq[2] + 0.1 * np.exp(-0.5 * d2 ** 2 / 9.0)
        kernel2_1 = 1.0 + 0.2 * xp[2] * xq[2] + 0.1 * np.exp(-0.5 * d2 ** 2 / 9.0)
        kernel2_2 = xp[2] * xq[2] + 0.1 * np.exp(-0.5 * d2 ** 2 / 9.0)
        # log(gamma) direction
        kernel3_1 = 1.0 + 0.2 * xp[3] * xq[3] + 0.1 * np.exp(-0.5 * d3 ** 2 / 9.0)
        kernel3_2 = xp[3] * xq[3] + 0.1 * np.exp(-0.5 * d3 ** 2 / 9.0)

        # log(rho_0) direction
        # idea here is to combine linear kernel on (rho0, gamma, zeta) for rho0 outside [-1,1] the tails
        #    using weaker linear functions in gamma, zeta directions (by + const)
        #    also the rho0 of course is actually bilinear
        # then use weak SE term to cover the oscillations in rho0 and in this example we use regular 
        #     linear kernels without a constant term because then it couples the gamma = 0 to high gamma data too much
        #     and messes up the mode switching zones
        #     put this in the paper.

        #* np.exp(-0.5 * (xp[3] + xq[3]) / 1.0) # 0.2, # * np.exp(-0.5 * (d1 ** 2 / (0.2 ** 2))
        kernel123 = (0.1 + soft_relu(-xp[1]) * soft_relu(-xq[1])) * kernel2_1 * kernel3_1 + \
            0.1 * np.exp(-0.5 * (d1 ** 2 / (0.2 + (xp[3] + xq[3]) / 2.0) ** 2)) * \
            soft_relu(1 - abs(xp[1])) * \
            soft_relu(1 - abs(xq[1])) * kernel2_2 * kernel3_2
        
        return kernel0 * kernel123

elif kernel_option == 2:
    def kernel(xp, xq, theta):
        # xp, xq are Nx1,Mx1 vectors (ln(xi), ln(rho_0), ln(1 + 10^3 * zeta), ln(1 + gamma))
        vec = xp - xq

        d1 = vec[1]  # first two entries
        d2 = vec[2]
        d3 = vec[3]

        # log(xi) direction
        kernel0 = 0.1 + xp[0] * xq[0]
        # log(rho_0) direction
        kernel1 = (
            soft_relu(-xp[1]) * soft_relu(-xq[1]) + 0.1 +
            0.1 #* np.exp(-0.5 * (xp[3] + xq[3]) / 1.0) # 0.2
            # * np.exp(-0.5 * (d1 ** 2 / (0.2 ** 2))
            * np.exp(-0.5 * (d1 ** 2 / (0.2 + (xp[3] + xq[3]) / 2.0) ** 2))
            * soft_relu(1 - abs(xp[1]))
            * soft_relu(1 - abs(xq[1]))
        )
        # log(zeta) direction
        #  kernel2 = xp[2] * xq[2] + 0.1 * np.exp(-0.5 * d2 ** 2 / 9.0)
        kernel2 = xp[2] * xq[2] + 0.1 * np.exp(-0.5 * d2 ** 2 / 9.0)
        # log(gamma) direction
        kernel3 = xp[3] * xq[3] + 0.1 * np.exp(-0.5 * d3 ** 2 / 9.0)
        return kernel0 * kernel1 * kernel2 * kernel3
    
elif kernel_option == 3:
    def kernel(xp, xq, theta):
        # xp, xq are Nx1,Mx1 vectors (ln(xi), ln(rho_0), ln(1 + 10^3 * zeta), ln(1 + gamma))
        vec = xp - xq

        d1 = vec[1]  # first two entries
        d2 = vec[2]
        d3 = vec[3]

        # log(xi) direction
        kernel0 = 1.0 + xp[0] * xq[0]
        # log(rho_0) direction
        kernel1_1 = soft_relu(-xp[1]) * soft_relu(-xq[1]) + 0.1
        kernel1_2 = (
            0.1 #* np.exp(-0.5 * (xp[3] + xq[3]) / 1.0) # 0.2
            # * np.exp(-0.5 * (d1 ** 2 / (0.2 ** 2))
            * np.exp(-0.5 * (d1 ** 2 / (0.2 + (xp[3] + xq[3]) / 2.0) ** 2))
            * soft_relu(1 - abs(xp[1]))
            * soft_relu(1 - abs(xq[1]))
        )
        # log(zeta) direction
        #  kernel2 = xp[2] * xq[2] + 0.1 * np.exp(-0.5 * d2 ** 2 / 9.0)
        kernel2 = (1.0 + 0.2 * xp[2] * xq[2])**2 + 0.1 * np.exp(-0.5 * d2 ** 2 / 9.0)
        # log(gamma) direction
        kernel3 = 1.0 + 0.5 * xp[3] * xq[3] + 0.1 * np.exp(-0.5 * d3 ** 2 / 9.0)
        return kernel1_1 * kernel0 * kernel2 * kernel3 #+ kernel1_2  # add perturbations separately
    
elif kernel_option == 4:
    # this one works great with the xi dimension and doesn't have any funky d/dgamma slopes becoming negative
    # at higher xi values

    def kernel(xp, xq, theta):
        # xp, xq are Nx1,Mx1 vectors (ln(xi), ln(rho_0), ln(1 + 10^3 * zeta), ln(1 + gamma))
        vec = xp - xq

        d1 = vec[1]  # first two entries
        d2 = vec[2]
        d3 = vec[3]

        # log(xi) direction
        kernel0 = 1.0 + xp[0] * xq[0]
        # log(rho_0) direction
        kernel1_1 = soft_relu(-xp[1]) * soft_relu(-xq[1]) + 0.1
        kernel1_2 = (
            0.1 #* np.exp(-0.5 * (xp[3] + xq[3]) / 1.0) # 0.2
            # * np.exp(-0.5 * (d1 ** 2 / (0.2 ** 2))
            * np.exp(-0.5 * (d1 ** 2 / (0.2 + (xp[3] + xq[3]) / 2.0) ** 2))
            * soft_relu(1 - abs(xp[1]))
            * soft_relu(1 - abs(xq[1]))
        )
        # log(zeta) direction
        #  kernel2 = xp[2] * xq[2] + 0.1 * np.exp(-0.5 * d2 ** 2 / 9.0)
        kernel2 = (1.0 + 0.2 * xp[2] * xq[2])**2 + 0.1 * np.exp(-0.5 * d2 ** 2 / 9.0)
        # log(gamma) direction
        kernel3 = 1.0 + 0.5 * xp[3] * xq[3] + 0.1 * np.exp(-0.5 * d3 ** 2 / 9.0)
        return kernel1_1 + kernel0 + kernel2 + kernel3 #+ kernel1_2  # add perturbations separately
    
elif kernel_option == 5:
    # didn't work that well and I think it's the multiplication
    # it was not related to the SE term on dimension 1
    def kernel(xp, xq, theta):
        # xp, xq are Nx1,Mx1 vectors (ln(xi), ln(rho_0), ln(1 + 10^3 * zeta), ln(1 + gamma))
        vec = xp - xq

        d1 = vec[1]  # first two entries
        d2 = vec[2]
        d3 = vec[3]

        # log(xi) direction
        kernel0 = 0.1 + xp[0] * xq[0]
        # log(rho_0) direction
        kernel1 = soft_relu(-xp[1]) * soft_relu(-xq[1]) + 0.1

        kernel1_2 = (
            1.0 + 0.03
            * np.exp(-0.5 * (d1 ** 2 / (0.2 + (xp[3] + xq[3]) / 2.0) ** 2))
            * soft_relu(1 - abs(xp[1]))
            * soft_relu(1 - abs(xq[1])) #* np.exp(-0.5 * d3 ** 2 / 9.0)
        )
        
        # log(zeta) direction
        #  kernel2 = xp[2] * xq[2] + 0.1 * np.exp(-0.5 * d2 ** 2 / 9.0)
        kernel2 = xp[2] * xq[2] + 0.1 * np.exp(-0.5 * d2 ** 2 / 9.0)
        # log(gamma) direction
        kernel3 = xp[3] * xq[3] + 0.1 * np.exp(-0.5 * d3 ** 2 / 9.0)
        
        return kernel0 * kernel1 * kernel2 * kernel3 * kernel1_2
    
elif kernel_option == 6:
    # yeah that helps, this is a reasonable model => but there's too much oscillation from the SE kernel term
    # maybe try and add xi, gamma, zeta product to the kernel1_2 term
    def kernel(xp, xq, theta):
        # xp, xq are Nx1,Mx1 vectors (ln(xi), ln(rho_0), ln(1 + 10^3 * zeta), ln(1 + gamma))
        vec = xp - xq

        d1 = vec[1]  # first two entries
        d2 = vec[2]
        d3 = vec[3]

        # log(xi) direction
        kernel0 = 1.0 + xp[0] * xq[0]
        # log(rho_0) direction
        kernel1_1 = soft_relu(-xp[1]) * soft_relu(-xq[1]) + 0.1

        kernel1_2 = (
            1.0 + 0.01
            * np.exp(-0.5 * (d1 ** 2 / (0.2 + (xp[3] + xq[3]) / 2.0) ** 2))
            * soft_relu(1 - abs(xp[1]))
            * soft_relu(1 - abs(xq[1])) #* np.exp(-0.5 * d3 ** 2 / 9.0)
        )
        
        # log(zeta) direction
        #  kernel2 = xp[2] * xq[2] + 0.1 * np.exp(-0.5 * d2 ** 2 / 9.0)
        kernel2 = (1.0 + 0.2 * xp[2] * xq[2])**2 + 0.1 * np.exp(-0.5 * d2 ** 2 / 9.0)
        # log(gamma) direction
        kernel3 = 1.0 + 0.5 * xp[3] * xq[3] + 0.1 * np.exp(-0.5 * d3 ** 2 / 9.0)
        
        return (kernel0 + kernel1_1 + kernel2 + kernel3) * kernel1_2
    
elif kernel_option == 7:
    # This one worked great! Captured more of the local data with a tuned value of the
    # SE coefficient to 0.02 and the linear terms are of a different style from the 
    # bilinear and SE part of rho0. 
    def kernel(xp, xq, theta):
        # xp, xq are Nx1,Mx1 vectors (ln(xi), ln(rho_0), ln(1 + 10^3 * zeta), ln(1 + gamma))
        vec = xp - xq

        d1 = vec[1]  # first two entries
        d2 = vec[2]
        d3 = vec[3]

        # log(xi) direction
        kernel0 = 1.0 + xp[0] * xq[0]
        # log(rho_0) direction
        kernel1_1 = soft_relu(-xp[1]) * soft_relu(-xq[1]) + 0.1

        kernel1_2 = (
            1.0 + 0.02
            * np.exp(-0.5 * (d1 ** 2 / 0.2**2 ))
            * soft_relu(1 - abs(xp[1]))
            * soft_relu(1 - abs(xq[1])) #* np.exp(-0.5 * d3 ** 2 / 9.0)
        )
        
        # log(zeta) direction
        #  kernel2 = xp[2] * xq[2] + 0.1 * np.exp(-0.5 * d2 ** 2 / 9.0)
        kernel2 = (1.0 + 0.2 * xp[2] * xq[2])**2 + 0.1 * np.exp(-0.5 * d2 ** 2 / 9.0)
        # log(gamma) direction
        kernel3 = 1.0 + 0.5 * xp[3] * xq[3] + 0.1 * np.exp(-0.5 * d3 ** 2 / 9.0)
        
        return kernel1_1 * (kernel0 + kernel2 + kernel3) + kernel1_2 * kernel0 * kernel2 * kernel3
    
elif kernel_option == 8:
    # This one worked great! Captured more of the local data with a tuned value of the
    # SE coefficient to 0.02 and the linear terms are of a different style from the 
    # bilinear and SE part of rho0. 
    def kernel(xp, xq, theta):
        # xp, xq are Nx1,Mx1 vectors (ln(xi), ln(rho_0), ln(1 + 10^3 * zeta), ln(1 + gamma))
        vec = xp - xq

        d1 = vec[1]  # first two entries
        d2 = vec[2]
        d3 = vec[3]

        # log(xi) direction
        kernel0 = 1.0 + xp[0] * xq[0]
        # log(rho_0) direction
        kernel1_1 = soft_relu(-xp[1]) * soft_relu(-xq[1]) + 0.1

        kernel1_2 = (
            1.0 + 0.02
            * np.exp(-0.5 * (d1 ** 2 / 0.2**2 ))
            * soft_relu(1 - abs(xp[1]))
            * soft_relu(1 - abs(xq[1])) #* np.exp(-0.5 * d3 ** 2 / 9.0)
        )
        
        # log(zeta) direction
        #  kernel2 = xp[2] * xq[2] + 0.1 * np.exp(-0.5 * d2 ** 2 / 9.0)
        kernel2 = (1.0 + 0.2 * xp[2] * xq[2])**2 + 0.1 * np.exp(-0.5 * d2 ** 2 / 9.0)
        # log(gamma) direction
        kernel3 = 1.0 + 0.5 * xp[3] * xq[3] + 0.1 * np.exp(-0.5 * d3 ** 2 / 9.0)
        
        return kernel1_1 * (kernel0 + kernel2 + kernel3) + kernel1_2 * kernel0 * kernel2 * kernel3

elif kernel_option == 9:
    # This one worked great! Captured more of the local data with a tuned value of the
    # SE coefficient to 0.02 and the linear terms are of a different style from the 
    # bilinear and SE part of rho0. 
    def kernel(xp, xq, theta, debug=False):
        # xp, xq are Nx1,Mx1 vectors (ln(1+xi), ln(rho_0), ln(1 + 10^3 * zeta), ln(1 + gamma))
        vec = xp - xq

        d1 = vec[1]  # first two entries
        d2 = vec[2]
        d3 = vec[3]

        BL_kernel = soft_relu(-xp[1]) * soft_relu(-xq[1]) + 0.1
        if debug: print(f"BL_kernel = {BL_kernel}")
        SE_kernel = (
            0.02
            * np.exp(-0.5 * (d1 ** 2 / 0.2**2 ))
            * soft_relu(1 - soft_abs(xp[1]))
            * soft_relu(1 - soft_abs(xq[1])) #* np.exp(-0.5 * d3 ** 2 / 9.0)
        )
        if debug: print(f"SE kernel = {SE_kernel}")
        gamma_kernel = 1.0 + 0.1 * xp[3] * xq[3]
        if debug: print(f"gamma vals = {xp[3]}, {xq[3]}")
        if debug: print(f"gamma kernel = {gamma_kernel}")
        xi_kernel = 0.1 * xp[0] * xq[0]
        if debug: print(f"xi kernel = {xi_kernel}")

        inner_kernel = BL_kernel * (1.0 + 0.1 * xp[3] * xq[3]) + SE_kernel + 0.1 * xp[0] * xq[0]
        if debug: print(f"inner kernel = {inner_kernel}")
        return inner_kernel * (1 + 0.01 * xp[2] * xq[2])
    
elif kernel_option == 10:
    # want to get lower RMSE
    # this is latest one as of 8/22/2024
    # zeta 

    def kernel(xp, xq, theta, debug=False):
        # xp, xq are Nx1,Mx1 vectors (ln(1+xi), ln(rho_0), ln(1 + 10^3 * zeta), ln(1 + gamma))
        vec = xp - xq

        d1 = vec[1]  # first two entries
        d2 = vec[2]
        d3 = vec[3]

        BL_kernel = soft_relu(-xp[1]) * soft_relu(-xq[1]) + 0.1
        if debug: print(f"BL_kernel = {BL_kernel}")
        # 0.02 was factor here before
        SE_factor = 0.05 * np.exp(-0.5 * (d1**2 / 0.2**2 + d3**2 / 0.3**2))
        SE_kernel = (
            SE_factor
            * soft_relu(1 - soft_abs(xp[1]))
            * soft_relu(1 - soft_abs(xq[1])) #* np.exp(-0.5 * d3 ** 2 / 9.0)
            * soft_relu(0.3 - xp[3])
            * soft_relu(0.3 - xq[3]) # correlate only low values of gamma together
        )
        if debug: print(f"SE kernel = {SE_kernel}")
        gamma_kernel = 1.0 + 0.1 * xp[3] * xq[3]
        if debug: print(f"gamma vals = {xp[3]}, {xq[3]}")
        if debug: print(f"gamma kernel = {gamma_kernel}")
        xi_kernel = 0.1 * xp[0] * xq[0]
        if debug: print(f"xi kernel = {xi_kernel}")

        # now have quadratic gamma kernel
        inner_kernel = BL_kernel * (1.0 + 0.1 * xp[3] * xq[3])**2 + SE_kernel + 0.1 * xp[0] * xq[0] + 0.02 * xp[2] * xq[2]
        if debug: print(f"inner kernel = {inner_kernel}")
        return inner_kernel

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
xi_bins = [[0.2, 0.4], [0.4, 0.6], [0.6, 0.8], [0.8, 1.0]]
rho0_bins = [[-2.5, -1.0], [-1.0, 0.0], [0.0, 1.0], [1.0, 2.5]]
zeta_bins = [[0.0, 0.1], [0.1, 0.5], [0.5, 1.0], [1.0, 2.0]]
gamma_bins = [[0.0, 0.1], [0.1, 1.0], [1.0, 3.0], [3.0, 5.0] ]

# randomly permute the arrays
rand_perm = np.random.permutation(N_data)
X = X[rand_perm, :]
Y = Y[rand_perm, :]

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


# 2d plots
_plot_2d = True
_plot_gamma = _plot_2d
_plot_zeta = _plot_2d
_plot_xi = _plot_2d
# 3d plots
_plot_3d = True
_plot_3d_gamma = _plot_3d
_plot_3d_xi = _plot_3d
_plot_3d_zeta = _plot_3d #_plot_3d

def closed_form_resid(x,y):
    if args.load == "Nx":
        xi = np.exp(x[0]) - 1.0
        rho0 = np.exp(x[1])
        zeta = (np.exp(x[2])-1.0)/1000.0
        gamma = np.exp(x[3]) - 1.0
        N11cr = 10000.0
        for m in range(1,51):
            for n in range(1,11):
                N11 = m**2 / rho0**2 * (1.0 + gamma) + rho0**2 * n**4 / m**2 + 2.0 * xi * n**2
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
        #plots_folder,
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

# test out a certain input
# debugging
xi = 0.9487
rho0 = 0.3121
gamma = 5.868
zeta = 0.0035
_Xtest = np.zeros((1,4))
_Xtest[0,0] = np.log(1.0+xi)
_Xtest[0,1] = np.log(rho0)
_Xtest[0,2] = np.log(1.0+1000.0*zeta)
_Xtest[0,3] = np.log(1.0+gamma)
#_Xtest[0,3] = np.log(1.0+1000.0*zeta)
print(f"Xtest = {_Xtest}")
#Ktest = np.array([[kernel(X_train[i,:], _Xtest[j,:], theta0, debug=True) for i in range(n_train)] for j in range(1)])
#print(f"Ktest = {Ktest}")
#for itrain in range(n_train):
#    print(f"mkernel = {Ktest[0,itrain]}")
#    print(f"alpha = {alpha[itrain]}")
#pred = Ktest @ alpha
#print(f"pred = {pred}")
#exit()

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

                print(f"zeta {izeta} in {zeta_bin[0]}, {zeta_bin[1]}")

                plt.figure(f"xi = {avg_xi:.2f}, zeta = {avg_zeta:.2f}", figsize=(8,6))

                colors = plt.cm.jet(np.linspace(0.0, 1.0, len(gamma_bins)))

                for igamma,gamma_bin in enumerate(gamma_bins[::-1]):

                    gamma_mask = np.logical_and(gamma_bin[0] <= X[:,3], X[:,3] <= gamma_bin[1])
                    avg_gamma = 0.5 * (gamma_bin[0] + gamma_bin[1])
                    mask = np.logical_and(xi_zeta_mask, gamma_mask)

                    #if np.sum(mask) == 0: continue

                    X_in_range = X[mask,:]
                    Y_in_range = Y[mask,:]
                    if args.resid:
                        Y_resid = np.array([closed_form_resid(X_in_range[i,:],Y_in_range[i,0]) for i in range(X_in_range.shape[0])])
                        Y_in_range = Y_resid.reshape((X_in_range.shape[0],1))

                    if np.sum(mask) != 0:
                        plt.plot(
                            X_in_range[:,1],
                            Y_in_range[:,0],
                            "o",
                            color=colors[igamma],
                            zorder=1+igamma,
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

                    if args.resid:
                        f_resid = np.array([closed_form_resid(X_plot[i,:],f_plot[i]) for i in range(X_plot.shape[0])])
                        f_plot = f_resid.reshape((X_plot.shape[0],1))

                    plt.plot(
                        rho0_vec,
                        f_plot,
                        "--",
                        color=colors[igamma],
                        zorder=1,
                        label=r"$\log(1+\gamma)" + f"\ in\ [{gamma_bin[0]:.1f},{gamma_bin[1]:.1f}" + r"]$",
                    )

                plt.legend()
                plt.xlabel(r"$\log{\rho_0}$")
                if args.load == "Nx":
                    plt.ylabel(r"$\log(N_{11,cr}^*)$")
                else:
                    plt.ylabel(r"$\log(N_{12,cr}^*)$")

                plt.savefig(os.path.join(GP_folder, f"2d-gamma-model_xi{ixi}_zeta{izeta}.png"), dpi=400)
                plt.close(f"xi = {avg_xi:.2f}, zeta = {avg_zeta:.2f}")

    if _plot_zeta:
        
        print(f"start 2d zeta plots...")

        for ixi, xi_bin in enumerate(xi_bins):
            xi_mask = np.logical_and(xi_bin[0] <= X[:,0], X[:,0] <= xi_bin[1])
            avg_xi = 0.5 * (xi_bin[0] + xi_bin[1])

            for igamma,gamma_bin in enumerate(gamma_bins):
                gamma_mask = np.logical_and(gamma_bin[0] <= X[:,3], X[:,3] <= gamma_bin[1])
                avg_gamma = 0.5 * (gamma_bin[0] + gamma_bin[1])
                xi_gamma_mask = np.logical_and(xi_mask, gamma_mask)

                for izeta, zeta_bin in enumerate(zeta_bins):
                    zeta_mask = np.logical_and(zeta_bin[0] <= X[:,2], X[:,2] <= zeta_bin[1])
                    avg_zeta = 0.5 * (zeta_bin[0] + zeta_bin[1])
                    mask = np.logical_and(xi_gamma_mask, zeta_mask)

                    #if np.sum(mask) == 0: continue

                    plt.figure(f"xi = {avg_xi:.2f}, gamma = {avg_gamma:.2f}", figsize=(8,6))

                    colors = plt.cm.jet(np.linspace(0.0, 1.0, len(zeta_bins)))

                    #if np.sum(mask) == 0: continue

                    X_in_range = X[mask,:]
                    Y_in_range = Y[mask,:]
                    if args.resid:
                        Y_resid = np.array([closed_form_resid(X_in_range[i,:],Y_in_range[i,0]) for i in range(X_in_range.shape[0])])
                        Y_in_range = Y_resid.reshape((X_in_range.shape[0],1))

                    if np.sum(mask) != 0:
                        plt.plot(
                            X_in_range[:,1],
                            Y_in_range[:,0],
                            "o",
                            color=colors[izeta],
                            zorder=1+izeta,
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

                    if args.resid:
                        f_resid = np.array([closed_form_resid(X_plot[i,:],f_plot[i]) for i in range(X_plot.shape[0])])
                        f_plot = f_resid.reshape((X_plot.shape[0],1))

                    plt.plot(
                        rho0_vec,
                        f_plot,
                        "--",
                        color=colors[izeta],
                        zorder=1,
                        label=r"$\log(1+10^3\zeta)" + f"\ in\ [{zeta_bin[0]:.1f},{zeta_bin[1]:.1f}" + r"]$"
                    )

                plt.legend()
                plt.xlabel(r"$\log{\rho_0}$")
                if args.load == "Nx":
                    plt.ylabel(r"$\log(N_{11,cr}^*)$")
                else:
                    plt.ylabel(r"$\log(N_{12,cr}^*)$")

                plt.savefig(os.path.join(GP_folder, f"2d-zeta-model_xi{ixi}_gamma{igamma}.png"), dpi=400)
                plt.close(f"xi = {avg_xi:.2f}, gamma = {avg_gamma:.2f}")  

    if _plot_xi:
        
        print(f"start 2d zeta plots...")

        for igamma,gamma_bin in enumerate(gamma_bins):
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

                    #if np.sum(mask) == 0: continue

                    plt.figure(f"zeta = {avg_zeta:.2f}, gamma = {avg_gamma:.2f}", figsize=(8,6))

                    colors = plt.cm.jet(np.linspace(0.0, 1.0, len(xi_bins)))

            
                    #if np.sum(mask) == 0: continue

                    X_in_range = X[mask,:]
                    Y_in_range = Y[mask,:]
                    if args.resid:
                        Y_resid = np.array([closed_form_resid(X_in_range[i,:],Y_in_range[i,0]) for i in range(X_in_range.shape[0])])
                        Y_in_range = Y_resid.reshape((X_in_range.shape[0],1))

                    if np.sum(mask) != 0:
                        plt.plot(
                            X_in_range[:,1],
                            Y_in_range[:,0],
                            "o",
                            color=colors[ixi],
                            zorder=1+ixi,
                        )

                    # plot the model now
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
                    if args.resid:
                        f_resid = np.array([closed_form_resid(X_plot[i,:],f_plot[i]) for i in range(X_plot.shape[0])])
                        f_plot = f_resid.reshape((X_plot.shape[0],1))

                    plt.plot(
                        rho0_vec,
                        f_plot,
                        "--",
                        color=colors[ixi],
                        zorder=1,
                        label=r"$\log(1+\xi)" f"\ in\ [{xi_bin[0]:.1f},{xi_bin[1]:.1f}" + r"]$"
                    )
                    

                plt.legend()
                plt.xlabel(r"$\log{\rho_0}$")
                if args.load == "Nx":
                    plt.ylabel(r"$\log(N_{11,cr}^*)$")
                else:
                    plt.ylabel(r"$\log(N_{12,cr}^*)$")

                plt.savefig(os.path.join(GP_folder, f"2d-xi-model_zeta{izeta}_gamma{igamma}.png"), dpi=400)
                plt.close(f"zeta = {avg_zeta:.2f}, gamma = {avg_gamma:.2f}")  

    if _plot_3d_gamma:

        # 3d plot of rho_0, gamma, lam_star for a particular xi and zeta range
        xi_bin = [0.2, 0.4]
        xi_mask = np.logical_and(xi_bin[0] <= X[:,0], X[:,0] <= xi_bin[1])
        avg_xi = 0.5 * (xi_bin[0] + xi_bin[1])

        zeta_bin = [0, 1]
        zeta_mask = np.logical_and(zeta_bin[0] <= X[:,2], X[:,2] <= zeta_bin[1])
        avg_zeta = 0.5 * (zeta_bin[0] + zeta_bin[1])
        xi_zeta_mask = np.logical_and(xi_mask, zeta_mask)

        plt.figure(f"3d rho_0, gamma, lam_star")
        ax = plt.axes(projection="3d", computed_zorder=False)

        colors = plt.cm.jet(np.linspace(0.0, 1.0, len(gamma_bins)))

        for igamma,gamma_bin in enumerate(gamma_bins):

            gamma_mask = np.logical_and(gamma_bin[0] <= X[:,3], X[:,3] <= gamma_bin[1])
            mask = np.logical_and(xi_zeta_mask, gamma_mask)

            X_in_range = X[mask,:]
            Y_in_range = Y[mask,:]
            if args.resid:
                Y_resid = np.array([closed_form_resid(X_in_range[i,:],Y_in_range[i,0]) for i in range(X_in_range.shape[0])])
                Y_in_range = Y_resid.reshape((X_in_range.shape[0],1))

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

        if args.resid:
            f_resid = np.array([closed_form_resid(X_plot[i,:],f_plot[i]) for i in range(X_plot.shape[0])])
            f_plot = f_resid.reshape((X_plot.shape[0],1))

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
        # plt.show()
        plt.savefig(os.path.join(GP_folder, f"gamma-3d.png"), dpi=400)
        plt.close(f"3d rho_0, gamma, lam_star")

    if _plot_3d_xi:

        # 3d plot of rho_0, gamma, lam_star for a particular xi and zeta range
        gamma_bin = [0.0, 0.1]
        gamma_mask = np.logical_and(gamma_bin[0] <= X[:,3], X[:,3] <= gamma_bin[1])
        avg_gamma = 0.5 * (gamma_bin[0] + gamma_bin[1])

        zeta_bin = [0.1, 0.5]
        zeta_mask = np.logical_and(zeta_bin[0] <= X[:,2], X[:,2] <= zeta_bin[1])
        avg_zeta = 0.5 * (zeta_bin[0] + zeta_bin[1])
        gamma_zeta_mask = np.logical_and(gamma_mask, zeta_mask)

        plt.figure(f"3d rho_0, xi, lam_star")
        ax = plt.axes(projection="3d", computed_zorder=False)

        colors = plt.cm.jet(np.linspace(0.0, 1.0, len(xi_bins)))

        for ixi, xi_bin in enumerate(xi_bins):

            xi_mask = np.logical_and(xi_bin[0] <= X[:,0], X[:,0] <= xi_bin[1])
            avg_xi = 0.5 * (xi_bin[0] + xi_bin[1])
            mask = np.logical_and(gamma_zeta_mask, xi_mask)

            X_in_range = X[mask,:]
            Y_in_range = Y[mask,:]
            if args.resid:
                Y_resid = np.array([closed_form_resid(X_in_range[i,:],Y_in_range[i,0]) for i in range(X_in_range.shape[0])])
                Y_in_range = Y_resid.reshape((X_in_range.shape[0],1))

            #print(f"X in range = {X_in_range}")
            #print(f"Y in range = {Y_in_range}")


            ax.scatter(
                X_in_range[:,0],
                X_in_range[:,1],
                Y_in_range[:,0],
                s=20,
                color=colors[ixi],
                edgecolors="black",
                zorder=2+ixi
            )

        # plot the scatter plot
        n_plot = 3000
        X_plot_mesh = np.zeros((30, 100))
        X_plot = np.zeros((n_plot, 4))
        ct = 0
        xi_vec = np.linspace(0.2, 1.0, 30)
        AR_vec = np.log(np.linspace(0.1, 10.0, 100))
        for ixi in range(30):
            for iAR in range(100):
                X_plot[ct, :] = np.array(
                    [xi_vec[ixi], AR_vec[iAR], avg_zeta, avg_gamma]
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
        if args.resid:
            f_resid = np.array([closed_form_resid(X_plot[i,:],f_plot[i]) for i in range(X_plot.shape[0])])
            f_plot = f_resid.reshape((X_plot.shape[0],1))

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
        ax.set_xlabel(r"$\log(1+\xi)$")
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
        # plt.show()
        plt.savefig(os.path.join(GP_folder, f"xi-3d.png"), dpi=400)
        plt.close(f"3d rho_0, xi, lam_star")

    if _plot_3d_zeta:

        # 3d plot of rho_0, gamma, lam_star for a particular xi and zeta range
        gamma_bin = [0.0, 0.1]
        gamma_mask = np.logical_and(gamma_bin[0] <= X[:,3], X[:,3] <= gamma_bin[1])
        avg_gamma = 0.5 * (gamma_bin[0] + gamma_bin[1])

        xi_bin = [0.2, 0.4]
        xi_mask = np.logical_and(xi_bin[0] <= X[:,0], X[:,0] <= xi_bin[1])
        avg_xi = 0.5 * (xi_bin[0] + xi_bin[1])

        gamma_xi_mask = np.logical_and(gamma_mask, xi_mask)

        plt.figure(f"3d rho_0, zeta, lam_star")
        ax = plt.axes(projection="3d", computed_zorder=False)

        colors = plt.cm.jet(np.linspace(0.0, 1.0, len(zeta_bins)))

        for izeta, zeta_bin in enumerate(zeta_bins):

            zeta_mask = np.logical_and(zeta_bin[0] <= X[:,2], X[:,2] <= zeta_bin[1])
            avg_zeta = 0.5 * (zeta_bin[0] + zeta_bin[1])

            mask = np.logical_and(gamma_xi_mask, zeta_mask)

            X_in_range = X[mask,:]
            Y_in_range = Y[mask,:]
            if args.resid:
                Y_resid = np.array([closed_form_resid(X_in_range[i,:],Y_in_range[i,0]) for i in range(X_in_range.shape[0])])
                Y_in_range = Y_resid.reshape((X_in_range.shape[0],1))

            #print(f"X in range = {X_in_range}")
            #print(f"Y in range = {Y_in_range}")


            ax.scatter(
                X_in_range[:,2],
                X_in_range[:,1],
                Y_in_range[:,0],
                s=20,
                color=colors[izeta],
                edgecolors="black",
                zorder=2+izeta
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
        if args.resid:
            f_resid = np.array([closed_form_resid(X_plot[i,:],f_plot[i]) for i in range(X_plot.shape[0])])
            f_plot = f_resid.reshape((X_plot.shape[0],1))

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
        #ax.set_zlim3d(0.0, np.log(50.0))
        #ax.set_zlim3d(1.0, 3.0)
        ax.view_init(elev=20, azim=20, roll=0)
        plt.gca().invert_xaxis()
        # plt.title(f"")
        # plt.show()
        plt.savefig(os.path.join(GP_folder, f"zeta-3d.png"), dpi=400)
        plt.close(f"3d rho_0, zeta, lam_star")

# only eval relative error on test set for zeta < 1
# because based on the model plots it appears that the patterns break down some for that
zeta_mask = X_test[:,2] < 1.0
X_test = X_test[zeta_mask, :]
Y_test = Y_test[zeta_mask, :]
n_test = X_test.shape[0]

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


if args.archive:
    # archive the data to the format of the 
    filename = "axialGP.csv" if args.load == "Nx" else "shearGP.csv"
    output_csv = "../archived_models/" + filename

    # remove the previous csv file if it exits
    # assume on serial here
    if os.path.exists(output_csv):
        os.remove(output_csv)


    # [log(1+xi), log(rho0), log(1+gamma), log(1+10^3 * zeta)]
    dataframe_dict = {
        "log(1+xi)" : X_train[:,0],
        "log(rho0)" : X_train[:,1],
        "log(1+gamma)" : X_train[:,3],
        "log(1+10^3*zeta)" : X_train[:,2], # gamma,zeta are flipped to the order used in TACS
        "alpha" : alpha[:,0],
    }
    model_df = pd.DataFrame(dataframe_dict)
    model_df.to_csv(output_csv)
