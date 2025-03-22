# demo a squared exponential GP on the axial dataset, measure interp and extrap error
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
import tensorflow as tf

sys.path.append("src/")
from closed_form_dataset import get_closed_form_data, split_data, nan_extrap_data
from plot_utils import plot_surface
from kernel_library import *

# inputs:
# ----------------------

# argparse
parser = argparse.ArgumentParser(description="Parse command-line arguments for data processing.")

# Boolean flags
parser.add_argument("--axial", action="store_false", default=True, help="Enable axial mode (default: True)")
parser.add_argument("--affine", action="store_false", default=True, help="Enable affine transformations (default: True)")
parser.add_argument("--log", action="store_false", default=True, help="Enable logging (default: True)")
parser.add_argument("--random", action="store_false", default=True, help="Enable random mode (default: True)")
parser.add_argument("--kernel", type=int, default=1, help="1 - SE, 2 - rational quadratic, 3 - Matern 3/2, 4 - Matern 5/2")
parser.add_argument("--ks", type=float, default=None, help="shear ks param, None if off, 1.0 good if on")

# Seed (only used if random is disabled)
parser.add_argument("--seed", type=int, default=1234, help="Seed for reproducibility (only used if --random is False)")
# parser.add_argument("--sigma_n", type=float, default=1e-4, help="Noise standard deviation (default: 1e-4)")

args = parser.parse_args()

# hyperparmeters
# -----------------------------------------------

# sigma_n = 1e-2
sigma_n = 1e-2
if args.kernel == 1:
    kernel = SE_kernel
    th = np.array([1.0, 8.0, 4.0])
elif args.kernel == 2:
    kernel = rational_quadratic_kernel
    th = np.array([3.0, 3.0, 1.0, 2.0])
elif args.kernel == 3:
    kernel = matern_3_2_kernel
    th = np.array([1.0, 3.0])
elif args.kernel == 4:
    kernel = matern_5_2_kernel
    th = np.array([1.0, 3.0])
elif args.kernel == 5:
    kernel = custom_kernel1
    # V1: 0.087 axial, 0.5 shear (smoothed or not)
    th = np.array([1.0, 8.0, 4.0, 1.0, 1.0, 0.1])

    # V2: 
    # th = np.array([8.0, 8.0, 4.0, 0.5, 1.0, 0.1])

    # V3: trying to remove gamma linear term 
    # th = np.array([8.0, 8.0, 4.0, 0.5, 0.0, 0.1])


# really should use a commercial hyperparameter optimize



# ---------------------------------------------------------

# If random is enabled, ignore the seed
if args.random:
    args.seed = None  # Seed has no effect in random mode

# get vars out of argparse
axial = args.axial; affine = args.affine; log = args.log

# get dataset:
# ------------------------------

# if random is True then seed is None and it changes each time
np.random.seed(args.seed)

# get only interp data
X, Y = get_closed_form_data(
    axial=axial, include_extrapolation=False, 
    affine_transform=affine, 
    log_transform=log,
    # no need for nan_interp, nan_extrap inputs yet that's for plotting
    n_rho0=20, n_gamma=10, n_xi=5,
)

# train GP surrogate model
# ------------------------

# now split into train and test for interpolation zone
X_train, Y_train, X_test, Y_test = split_data(X, Y, train_test_split=0.9)

# fast way to compute kernel functions, from ml_pde_buckling final proj
x_train_L = tf.expand_dims(X_train, axis=1)
x_train_R = tf.expand_dims(X_train, axis=0)

nugget = sigma_n**2 * np.eye(x_train_L.shape[0])
K_train = kernel(x_train_L, x_train_R, th) + nugget
# print(f"{K_train.shape=}")
# plt.imshow(K_train)

# get the training weights
alpha_train = np.linalg.solve(K_train, Y_train)

# get the full plot dataset
# -------------------------
X_plot, Y_plot_truth = get_closed_form_data(
    axial=axial, include_extrapolation=True, 
    affine_transform=affine, 
    shear_ks_param=args.ks,
    log_transform=log,
    # no need for nan_interp, nan_extrap inputs yet that's for plotting
    n_rho0=20, n_gamma=10, n_xi=5,
)

# make predictions on the plot data
# ---------------------------------
x_plot_L = tf.expand_dims(X_plot, axis=1)
K_cross = kernel(x_plot_L, x_train_R, th)
Y_plot_pred = np.dot(K_cross, alpha_train)

# compute the interpolation error
# ------------------------------

x_test_L = tf.expand_dims(X_test, axis=1)
K_cross_test = kernel(x_test_L, x_train_R, th)
Y_test_pred = np.dot(K_cross_test, alpha_train)
sq_diff = (Y_test - Y_test_pred)**2
test_interp_RMSE = np.sqrt(np.mean(sq_diff))
print(f"{test_interp_RMSE=}")

# compute the extrapolation error
# -------------------------------

# get the extrapolation dataset
X_extrap, Y_extrap_truth = get_closed_form_data(
    axial=axial, 
    include_extrapolation=True,
    include_interpolation=False, 
    shear_ks_param=args.ks,
    affine_transform=affine, 
    log_transform=log,
    # no need for nan_interp, nan_extrap inputs yet that's for plotting
    n_rho0=20, n_gamma=10, n_xi=5,
)

x_extrap_L = tf.expand_dims(X_extrap, axis=1)
K_cross_extrap = kernel(x_extrap_L, x_train_R, th)
Y_extrap_pred = np.dot(K_cross_extrap, alpha_train)
sq_diff_extrap = (Y_extrap_truth - Y_extrap_pred)**2
test_extrap_RMSE = np.sqrt(np.mean(sq_diff_extrap))
print(f"{test_extrap_RMSE=}")

# plot the surrogate model surface
# --------------------------------

#    can we separately plot the interp and extrap parts of our data,
#    with manual nan on extrap vs interp regions
Y = Y_plot_pred
Y_plot_pred_extrap = nan_extrap_data(X_plot, Y, log=log, affine=affine,
                                     nan_extrap=False)
Y_plot_pred_interp = nan_extrap_data(X_plot, Y, log=log, affine=affine,
                                     nan_extrap=True)

# plot with gamma
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
colors = ["blue", "gray"]
for i,Yp in enumerate([Y_plot_pred_extrap, Y_plot_pred_interp]):
    plot_surface(X_plot, Y=Yp, 
        Y_truth=Y_plot_truth,
        surf_color_map=colors[i],
        var_exclude_ind=2, 
        nx1=20, nx2=10,
        var_exclude_range=[0.3]*2,
        ax=ax, show=False)
ax.set_zlim(0.0, 6.0)

fs = 14
fw = 'bold'
lp = 10

load_str = "axial" if axial else "shear"
affine_str = "affine" if affine else "not-affine"
log_str = "log" if log else "not-log"
plt.title(f"{load_str}, {affine_str}, {log_str}")

ax.set_xlabel(r"$\mathbf{\ln(\rho_0^*)}$" if affine else r"$\mathbf{\ln(\rho_0)}$", fontsize=fs, fontweight=fw, labelpad=lp)
ax.set_ylabel(r"$\mathbf{\ln(1+\gamma)}$", fontsize=fs, fontweight=fw, labelpad=lp)
ax.set_zlabel(r"$\mathbf{\ln(\overline{N}_{11}^{cr})}$" if axial else r"$\mathbf{\ln(\overline{N}_{12}^{cr})}$", fontsize=fs, fontweight=fw, labelpad=0)

plt.show()

# plot with xi
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
# colors = ["blue", "gray"]
# for i,Yp in enumerate([Y_plot_pred_extrap, Y_plot_pred_interp]):
#     plot_surface(X_plot, Y=Yp, 
#         Y_truth=Y_plot_truth,
#         surf_color_map=colors[i],
#         var_exclude_ind=1, 
#         nx1=20, nx2=5,
#         var_exclude_range=[0.0]*2,
#         ax=ax, show=False)
# ax.set_zlim(0.0, 6.0)
# plt.show()