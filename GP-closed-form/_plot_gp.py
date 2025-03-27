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
parser.add_argument("--kernel", type=int, default=5, help="1 - SE, 2 - rational quadratic, 3 - Matern 3/2, 4 - Matern 5/2")
parser.add_argument("--ks", type=float, default=None, help="shear ks param, None if off, 1.0 good if on")
parser.add_argument("--show", action="store_false", default=True, help="Enable logging (default: True)")

# Seed (only used if random is disabled)
parser.add_argument("--seed", type=int, default=1234, help="Seed for reproducibility (only used if --random is False)")
# parser.add_argument("--sigma_n", type=float, default=1e-4, help="Noise standard deviation (default: 1e-4)")

args = parser.parse_args()

# hyperparmeters
# -----------------------------------------------

# temporarily make xi only from 0.3 to 0.3 for easier fit check
# xi_bounds = [0.3]*2; nxi = 1
# and correct ranges
xi_bounds = [0.3, 1.5]; nxi = 5

# sigma_n = 1e-2
sigma_n = 1e-2
if args.kernel == 1:
    kernel = SE_kernel
    # th = np.array([1.0, 8.0, 4.0])
    # th = np.array([0.38, 6.6, 10.0])

    # prev best
    th = np.array([4.0, 8.0, 4.0])

    # k-fold optimized
    th = np.array([1.61690217, 6.434866, 10.0])
elif args.kernel == 2:
    kernel = rational_quadratic_kernel
    th = np.array([1.0, 2.0])
elif args.kernel == 3:
    kernel = matern_3_2_kernel
    # th = np.array([1.0, 3.0])
    th = np.array([1.0, 8.0])
elif args.kernel == 4:
    kernel = matern_5_2_kernel
    # th = np.array([1.0, 3.0])
    th = np.array([1.0, 8.0])
elif args.kernel == 5:
    # KERNEL 5 is the final kernel!!
    kernel = buckling_SE_kernel

    # V1: 0.087 axial, 0.5 shear (smoothed or not)
    # th = np.array([1, 8, 4, 1, 1, 0.1])
    th = np.array([1, 8, 4, 1, 1, 0.1] + [1.0])

    # axial, log optimal - from k-fold optimization
    # th = np.array([4.48755514e-01, 3.06074470e+00, 1.79802253e-01, 1.00000000e+01,
    #    1.00409476e+00, 9.90178467e-03, 3.15247290e+00])
    
    # th[-2] = 0.0
    # th[-1] = 1.0

elif args.kernel == 6:
    # buckling + RQ
    kernel = buckling_RQ_kernel # with RQ instead of SE

    th = np.array([5.0, 1.0, 1e-1, 1.0, 2.0, 1.0])
    # th = np.array([10.0, 0.9959462681018701, 0.048967409869886735, 2.0997981037015867, 0.36781754937098754, 0.9955409650507732])
    # th = np.array([10.0, 1.0217731489244601, 0.03255743866181401, 1.5527781674771741, 0.5156747043662844, 0.8814924462819974])
    # th = np.array([10.        ,  1.0208472 ,  0.0347678 ,  1.56025982,  0.51233982,
    #     0.1       ])
    th = np.array([10.        ,  1.05728864,  0.25339722,  7.24746841,  0.1       ,
        0.1       ])


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

print(f"{th=}")

# get only interp data
X, Y = get_closed_form_data(
    axial=axial, include_extrapolation=False, 
    affine_transform=affine, 
    log_transform=log,
    xi_bounds=xi_bounds,
    shear_ks_param=args.ks,
    # no need for nan_interp, nan_extrap inputs yet that's for plotting
    n_rho0=20, n_gamma=10, n_xi=nxi,
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
    xi_bounds=xi_bounds,
    # no need for nan_interp, nan_extrap inputs yet that's for plotting
    n_rho0=20, n_gamma=10, n_xi=nxi,
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
if args.log:
    sq_diff = (Y_test - Y_test_pred)**2
    Y_mean = np.mean(Y_test)
    sq_mean_diff = (Y_test_pred - Y_mean)**2
else: # not log
    sq_diff = (np.log(Y_test) - np.log(Y_test_pred))**2
    Y_log_mean = np.mean(np.log(Y_test))
    sq_mean_diff = (Y_log_mean - np.log(Y_test_pred))**2

test_interp_RMSE = np.sqrt(np.mean(sq_diff))
print(f"{test_interp_RMSE=}")

test_interp_Rsq = 1.0 - np.sum(sq_diff) / np.sum(sq_mean_diff)

# compute the extrapolation error
# -------------------------------

# get the extrapolation dataset
X_extrap, Y_extrap_truth = get_closed_form_data(
    axial=axial, 
    include_extrapolation=True,
    include_interpolation=False,
    xi_bounds=xi_bounds, 
    shear_ks_param=args.ks,
    affine_transform=affine, 
    log_transform=log,
    # no need for nan_interp, nan_extrap inputs yet that's for plotting
    n_rho0=20, n_gamma=10, n_xi=nxi,
)

x_extrap_L = tf.expand_dims(X_extrap, axis=1)
K_cross_extrap = kernel(x_extrap_L, x_train_R, th)
Y_extrap_pred = np.dot(K_cross_extrap, alpha_train)
if args.log:
    sq_diff_extrap = (Y_extrap_truth - Y_extrap_pred)**2
    mean_extrap = np.mean(Y_extrap_truth)
    sq_mean_diff_extrap = (Y_extrap_pred - mean_extrap)**2
else: # not log
    sq_diff_extrap = (np.log(Y_extrap_truth) - np.log(Y_extrap_pred))**2
    log_mean_extrap = np.mean(np.log(Y_extrap_truth))
    sq_mean_diff_extrap = (np.log(Y_extrap_pred) - log_mean_extrap)**2

test_extrap_RMSE = np.sqrt(np.mean(sq_diff_extrap))
print(f"{test_extrap_RMSE=}")

test_extrap_Rsq = 1.0 - np.sum(sq_diff_extrap) / np.sum(sq_mean_diff_extrap)

print("------------------")
print(f"{test_interp_Rsq=}")
print(f"{test_extrap_Rsq=}")

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
        # affine_shift=th[3] if args.kernel == 5 else None,
        surf_color_map=colors[i],
        var_exclude_ind=2, 
        nx1=20, nx2=10,
        var_exclude_range=[0.3]*2,
        ax=ax, show=False)

ax.set_zlim(0.0, 6.0)
# if args.log:
#     ax.set_zlim(0.0, 6.0)
# else:
#     ax.set_zlim(0.0, 50.0)

fs = 14
fw = 'bold'
lp = 10

load_str = "axial" if axial else "shear"
affine_str = "affine" if affine else "not-affine"
log_str = "log" if log else "not-log"
plt.title(f"{load_str}, {affine_str}, {log_str}")

ax.set_xlabel(r"$\mathbf{\ln(\rho_0^*)}$" if affine else r"$\mathbf{\ln(\rho_0)}$", fontsize=fs, fontweight=fw, labelpad=lp)
ax.set_ylabel(r"$\mathbf{\ln(1+\gamma)}$", fontsize=fs, fontweight=fw, labelpad=lp)
ax.set_zlabel(r"$\mathbf{\ln(\overline{N}_{11}^{cr})}$" if axial else r"$\mathbf{\ln(\overline{N}_{12}^{cr})}$", fontsize=fs, fontweight=fw, labelpad=6)

if args.show:
    plt.show()
else:
    axial_str = "axial" if args.axial else "shear"
    prefix = "ks_" if args.ks is not None else ""
    plt.savefig(f"kernel{args.kernel}_{prefix}{axial_str}.png")

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