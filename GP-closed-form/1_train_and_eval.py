# demo a squared exponential GP on the axial dataset, measure interp and extrap error
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import tensorflow as tf

sys.path.append("src/")
from closed_form_dataset import get_closed_form_data, split_data
from eval_GPs import eval_GPs
from plot_GPs import plot_GPs
from eval_utils import eval_Rsquared, eval_rmse
from hyperparam_opt import kfold_hyperparameter_optimization
from kernel_library import *


# argparse
# --------------------------------------
parser = argparse.ArgumentParser(description="Parse command-line arguments for data processing.")
parser.add_argument("--axial", action=argparse.BooleanOptionalAction, default=False, help="Enable or disable axial mode (default: False)")
parser.add_argument("--affine", action=argparse.BooleanOptionalAction, default=False, help="Enable or disable affine transformations (default: False)")
parser.add_argument("--log", action=argparse.BooleanOptionalAction, default=False, help="Enable or disable logging (default: False)")
parser.add_argument("--random", action="store_false", default=True, help="Enable random mode (default: True)")
parser.add_argument("--kfolds", type=int, default=20, help="number of kfolds")
parser.add_argument("--dataf", type=float, default=1.0, help='fraction of full dataset size (out of 1000) for doing less trials')
parser.add_argument("--ks", type=float, default=None, help="shear_ks_param")
parser.add_argument("--kernel", type=str, default='buckling+RQ', help="SE, matern-3/2, matern-5/2, RQ, buckling+SE, buckling+RQ")
parser.add_argument("--seed", type=int, default=1234, help="Seed for reproducibility (only used if --random is False)")
args = parser.parse_args()

# make the folder for this kernel
# -------------------------------
if not os.path.exists("output"):
    os.mkdir("output")

folder_name = f"output/{args.kernel}"
if not os.path.exists(folder_name):
    os.mkdir(folder_name)

if not os.path.exists(f"{folder_name}/opt"):
    os.mkdir(f"{folder_name}/opt")

# base name for each file
axial_str = "axial" if args.axial else "shear"
log_str = "log" if args.log else "nolog"
affine_str = "affine" if args.affine else "noaffine"
kfold_str = f"kfold{args.kfolds}"
ks_str = "" if args.ks is None else f"ks{args.ks}_"
base_name = f"{kfold_str}_{ks_str}{axial_str}_{affine_str}_{log_str}"

# make a file in this folder for this specific case
txt_file = f"{folder_name}/{base_name}.txt"
txt_hdl = open(txt_file, "w")

# kernels list here, setup for hyperparam opt
# --------------------------------------


n_data_mult = args.dataf**(1.0/3.0)
n_rho = int(20 * n_data_mult)
n_gamma = int(10 * n_data_mult)
n_xi = np.max([int(5 * n_data_mult), 3])


sigma_n = 1e-2 # just fixed this for now, not included in hyperparameter opt..

if args.kernel == "SE":
    kernel = SE_kernel
    # [l_rho0, l_gamma, l_xi]
    lbounds = np.array([0.1]*3)
    theta0 = np.array([1, 8, 4])
    ubounds = np.array([10.0]*3)

elif args.kernel == "matern-3_2":
    kernel = matern_3_2_kernel
    # [coeff, length]
    lbounds = np.array([0.1]*2)
    theta0 = np.array([1.0, 2.0])
    ubounds = np.array([10.0]*2)

elif args.kernel == "matern-5_2":
    kernel = matern_5_2_kernel
    # [coeff, length]
    lbounds = np.array([0.1]*2)
    theta0 = np.array([1.0, 2.0])
    ubounds = np.array([10.0]*2)

elif args.kernel == "RQ":
    kernel = rational_quadratic_kernel
    # [length, alpha]
    lbounds = np.array([0.1]*2)
    theta0 = np.array([3.0, 2.0])
    ubounds = np.array([10.0]*2)

elif args.kernel == "buckling+SE":
    kernel = buckling_SE_kernel
    # [L1, L2, L3, relu_alpha, gamma_coeff, SE_coeff, constant]
    lbounds = np.array([0.1]*3 + [1e-2] + [0.1, 1e-3, 0.1])
    ubounds = np.array([10.0]*3 + [10.0] + [10.0]*3)
    theta0 = np.array([1.0, 8.0, 4.0, 5.0, 1.0, 0.1, 0.1])


elif args.kernel == "buckling+RQ":
    kernel = buckling_RQ_kernel
    # [relu_alph, gamma_coeff, RQ_coeff, length, alpha, constant]
    # lbounds = np.array([0.1]*2 + [1e-3] + [0.1]*3)
    lbounds = np.array([0.1]*2 + [1e-3] + [0.1] + [1e-3]*2)
    ubounds = np.array([10.0]*6)
    theta0 = np.array([5.0, 1.0, 1e-1, 1.0, 2.0, 1.0])

else:
    raise AssertionError("args.kernel is not one of the allowed types..")


# get interpolation dataset
# -------------------------

X_interp, Y_interp = get_closed_form_data(
    axial=args.axial, 
    include_extrapolation=False, 
    affine_transform=args.affine, 
    log_transform=args.log,
    n_rho0=n_rho, n_gamma=n_gamma, n_xi=n_xi,
    # n_rho0=20, n_gamma=10, n_xi=5,
)

# run k-fold cross validation
# ---------------------------

import time
start_time = time.time()

theta_opt_dict = kfold_hyperparameter_optimization(
    kernel,
    lbounds,
    theta0,
    ubounds,
    num_kfolds=args.kfolds,
    X=X_interp, 
    Y=Y_interp,
    can_print=False,
    snopt_options={
        "Major iterations limit": 500,
        "Print file": f"{folder_name}/opt/{base_name}_SNOPT_print.out",
        "Summary file": f"{folder_name}/opt/{base_name}_SNOPT_summary.out",
    }
)

train_time_sec = time.time() - start_time
print(f"{train_time_sec=:.4e}")

theta_opt = np.array(theta_opt_dict['theta'])
print(f"{theta_opt=}")

txt_hdl.write("--------------------------------------------\n\n")
txt_hdl.write(f"hyperparameter optimization with kfolds={args.kfolds}\n")
txt_hdl.write("\ttheta_opt:\n")
txt_hdl.write(f"\t{theta_opt=}\n\n")
txt_hdl.write("--------------------------------------------\n\n")

# eval GPs metrics - R^2
# ---------------------------
interp_Rsq, extrap_Rsq = eval_GPs(
    kernel,
    theta_opt,
    n_trials=30,
    sigma_n=sigma_n,
    train_test_frac=0.8,
    shear_ks_param=None,
    axial=args.axial,
    affine=args.affine,
    log=args.log,
    # n_rho0=20, n_gamma=10, n_xi=5,
    n_rho0=n_rho, n_gamma=n_gamma, n_xi=n_xi,
    metric_func=eval_Rsquared,
    percentile=50.0,
)

# txt_hdl.write("-------------\n\n")
txt_hdl.write(f"R^2 metrics:\n")
txt_hdl.write(f"\t{interp_Rsq=}\n\n")
txt_hdl.write(f"\t{extrap_Rsq=}\n\n")
txt_hdl.write("--------------------------------------------\n\n")

# eval GPs metrics - RMSE
# ---------------------------
interp_rmse, extrap_rmse = eval_GPs(
    kernel,
    theta_opt,
    n_trials=30,
    sigma_n=sigma_n,
    train_test_frac=0.8,
    shear_ks_param=None,
    axial=args.axial,
    affine=args.affine,
    log=args.log,
    # n_rho0=20, n_gamma=10, n_xi=5,
    n_rho0=n_rho, n_gamma=n_gamma, n_xi=n_xi,
    metric_func=eval_rmse,
    percentile=50.0,
)

# txt_hdl.write("-------------\n\n")
txt_hdl.write(f"RMSE metrics:\n")
txt_hdl.write(f"\t{interp_rmse=}\n\n")
txt_hdl.write(f"\t{extrap_rmse=}\n\n")
txt_hdl.write("--------------------------------------------\n\n")


# train the model for plotting
# ----------------------------

# now split into train and test for interpolation zone
X_train, Y_train, X_test, Y_test = split_data(X_interp, Y_interp, train_test_split=0.9)
x_train_L = tf.expand_dims(X_train, axis=1)
x_train_R = tf.expand_dims(X_train, axis=0)

# compute kernel matrix and training weights
nugget = sigma_n**2 * np.eye(x_train_L.shape[0])
K_train = kernel(x_train_L, x_train_R, theta_opt) + nugget
alpha_train = np.linalg.solve(K_train, Y_train)

# make plot data predictions
# -------------------------

# get full dataset including extrapolation zone
X_plot, Y_plot = get_closed_form_data(
    axial=args.axial,
    include_extrapolation=True, 
    affine_transform=args.affine,
    log_transform=args.log,
    # n_rho0=20, n_gamma=10, n_xi=5,
    n_rho0=n_rho, n_gamma=n_gamma, n_xi=n_xi,
)

# make predictionss on plot data
x_plot_L = tf.expand_dims(X_plot, axis=1)
K_cross = kernel(x_plot_L, x_train_R, theta_opt)
Y_plot_pred = np.dot(K_cross, alpha_train)

# plot the GP
# -----------

plot_GPs(
    X_plot,
    Y_plot,
    Y_plot_pred,
    folder_name=folder_name,
    base_name=base_name,
    axial=args.axial,
    affine=args.affine,
    log=args.log,
    nx1=n_rho, nx2=n_gamma,
    show=False, # means that it saves file to png/svg
)

# cleanup
# -------

txt_hdl.close()


print(f"R^2 metrics:\n")
print(f"\t{interp_Rsq=}\n\n")
print(f"\t{extrap_Rsq=}\n\n")
print("--------------------------------------------\n\n")



print(f"{train_time_sec=:.4e}")