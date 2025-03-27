import sys
import pandas as pd
import numpy as np
import argparse
import tensorflow as tf
import os

sys.path.append("src/")
from kernel_library import *
from plot_utils import plot_3d_gamma, plot_3d_xi
from data_transforms import affine_transform
from hyperparam_opt import kfold_hyperparameter_optimization

# argparse
# --------

parent_parser = argparse.ArgumentParser(add_help=False)
parent_parser.add_argument("--load", type=str, default="Nx")
parent_parser.add_argument("--ntrain", type=int, default=3000)
parent_parser.add_argument("--opt", action=argparse.BooleanOptionalAction, default=True, help="Enable or disable axial mode (default: False)")
parent_parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=False, help="Enable or disable axial mode (default: False)")
args = parent_parser.parse_args()

# random seed (fixed for now..)
np.random.seed(123456)

# get the dataset
# ---------------

csv_filename = f"{args.load}_stiffened"
df = pd.read_csv("data/" + csv_filename + ".csv")

# extract only the model columns
X0 = df[["log(1+xi)", "log(rho_0)", "log(1+10^3*zeta)", "log(1+gamma)"]].to_numpy()
Y = df["log(eig_FEA)"].to_numpy()
Y = np.reshape(Y, newshape=(Y.shape[0], 1))

# shuffle and then select random subset of data
N = X0.shape[0]
my_ind = np.array([_ for _ in range(N)])
shuffled_ind = np.random.permutation(my_ind)
subset_ind = shuffled_ind[:args.ntrain]

# now get the subset of data
X0_train = X0[subset_ind,:]
Y_train = Y[subset_ind]

# affine transform
use_affine = args.load == "Nx" # only use for axial
# use_affine = False
if use_affine:
    X_train = affine_transform(X0_train, is_log=True)
    X = affine_transform(X0, is_log=True)
else: 
    X_train = X0_train
    X = X0

# set bounds of the hyperparam opt and initial
# --------------------------------------------

# hyperparms for the buckling + GP kernel
kernel = buckling_RQ_kernel
# [relu_alph, gamma_coeff, RQ_coeff, length, alpha, constant, log10(sigma_n)]
lbounds = np.array([0.1] + [1e-3] + [1e-3] + [0.1] + [1e-4]*2 + [-4])
theta0 = np.array([5.0, 1.0, 1e-1, 1.0, 2.0, 1.0] + [-2])
ubounds = np.array([10.0]*6 + [0])

if not args.opt:
    theta_opt_v1 = np.array([5.080335025828115, 0.1, 10.0, 0.7564161097004841, 1.6312811045492983, 0.0001, -1.0314949162668399])
    
    theta0 = theta_opt_v1

# prelim train the model for plotting
# -----------------------------------

# use whole dataset for train for now (can change later)
x_train_L = tf.expand_dims(X_train, axis=1)
x_train_R = tf.expand_dims(X_train, axis=0)
sigma_n = np.power(10.0, theta0[-1]) 
nugget_term = sigma_n**2 * np.eye(x_train_L.shape[0])
K_train0 = kernel(x_train_L, x_train_R, theta0) + nugget_term
alpha_train0 = np.linalg.solve(K_train0, Y_train)


# plot the initial model
# ----------------------

if not os.path.exists(f"plots/{args.load}_stiffened"):
    os.mkdir(f"plots/{args.load}_stiffened")

if not os.path.exists(f"plots/{args.load}_stiffened/GP"):
    os.mkdir(f"plots/{args.load}_stiffened/GP")

plot_3d_gamma(
    X=X, Y=Y,
    X_train=X_train,
    kernel_func=kernel,
    theta=theta0,
    alpha_train=alpha_train0,
    xi_bin=[0.5, 0.7],
    zeta_bin=[0.2, 0.4],
    folder_name=f"plots/{args.load}_stiffened/GP",
    load_name=args.load,
    file_prefix="init_",
    is_affine=use_affine,
    show=args.show,
)

print(f'{X0_train.shape} {X.shape=}')
# exit()

plot_3d_xi(
    X=X, Y=Y,
    X_train=X_train,
    kernel_func=kernel,
    theta=theta0,
    alpha_train=alpha_train0,
    gamma_bin=[0.0, 0.1],
    zeta_bin=[0.0, 1.0],
    folder_name=f"plots/{args.load}_stiffened/GP",
    load_name=args.load,
    file_prefix="init_",
    is_affine=use_affine,
    show=args.show,
)

# optimize 
# --------

if args.opt:

    # now we need to do kfold hyperparameter optimization
    # ---------------------------------------------------

    theta_opt_dict = kfold_hyperparameter_optimization(
        kernel,
        lbounds=lbounds,
        theta0=theta0,
        ubounds=ubounds,
        num_kfolds=10,
        X=X_train, Y=Y_train,
        snopt_options={
            "Major iterations limit": 500,
            # "Print file": f"{folder_name}/opt/{base_name}_SNOPT_print.out",
            # "Summary file": f"{folder_name}/opt/{base_name}_SNOPT_summary.out",
        },
        can_print=True,
    )

    theta_opt = np.array(theta_opt_dict['theta'])
    print(f"{theta_opt=}")

    # train the model again
    # ---------------------

    # use whole dataset for train for now (can change later)
    sigma_n = np.power(10.0, theta_opt[-1]) 
    nugget_term = sigma_n**2 * np.eye(x_train_L.shape[0])
    K_train_opt = kernel(x_train_L, x_train_R, theta_opt) + nugget_term
    alpha_train_opt = np.linalg.solve(K_train_opt, Y)

    # plot after hyperparam opt
    # -------------------------

    plot_3d_gamma(
        X=X, Y=Y,
        X_train=X_train,
        kernel_func=kernel,
        theta=theta0,
        alpha_train=alpha_train_opt,
        xi_bin=[0.5, 0.7],
        zeta_bin=[0.2, 0.4],
        folder_name=f"plots/{args.load}_stiffened/GP",
        load_name=args.load,
        file_prefix="opt_",
        is_affine=use_affine,
        show=args.show,
    )

    plot_3d_xi(
        X=X, Y=Y,
        X_train=X_train,
        kernel_func=kernel,
        theta=theta0,
        alpha_train=alpha_train_opt,
        gamma_bin=[0.0, 0.1],
        zeta_bin=[0.0, 1.0],
        folder_name=f"plots/{args.load}_stiffened/GP",
        load_name=args.load,
        file_prefix="opt_",
        is_affine=use_affine,
        show=args.show,
    )
