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
from eval_utils import *

# argparse
# --------

parent_parser = argparse.ArgumentParser(add_help=False)
parent_parser.add_argument("--load", type=str, default="Nx")
parent_parser.add_argument("--ntrain", type=int, default=1500) # 1000
parent_parser.add_argument("--kfold", type=int, default=20)
parent_parser.add_argument("--gammalb", type=int, default=1e-1)
parent_parser.add_argument("--opt", action=argparse.BooleanOptionalAction, default=True, help="Enable or disable axial mode (default: False)")
parent_parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=False, help="Enable or disable axial mode (default: False)")
args = parent_parser.parse_args()

# random seed (fixed for now..)
np.random.seed(123456)

# make folders
# ------------
if not os.path.exists("output"):
    os.mkdir("output")

if not os.path.exists("output/opt"):
    os.mkdir("output/opt")

axial_str = "axial" if args.load == "Nx" else "shear"
kfold_str = f"kfold{args.kfold}"
ntrain_str = f"ntrain{args.ntrain}"
gammalb_log10 = np.log10(args.gammalb)
gamma_lb_str = f"gammalb{args.gammalb}"
base_name = f"{axial_str}_{kfold_str}_{ntrain_str}_{gamma_lb_str}"

# make a file in this folder for this specific case
txt_file = f"output/{base_name}.txt"
txt_hdl = open(txt_file, "w")

# get the dataset
# ---------------

csv_filename = f"{args.load}_stiffened"
df = pd.read_csv("data/" + csv_filename + ".csv")

# extract only the model columns
X0 = df[["log(1+xi)", "log(rho_0)", "log(1+10^3*zeta)", "log(1+gamma)"]].to_numpy()
# fix order of data
reorder_data = [1, 0, 3, 2]
# now is order [log(rho_0), log(1+xi), log(1+gamma), log(1+10^3 * zeta)]
X0 = X0[:, np.array(reorder_data)]
# print(f"{X0=}")
# exit()

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

# test set
rem_ind = [_ for _ in range(N) if not(_ in subset_ind)]
X0_test = X0[rem_ind,:]
Y_test = Y[rem_ind]

# affine transform
use_affine = args.load == "Nx" # only use for axial
# use_affine = False
if use_affine:
    X_train = affine_transform(X0_train, is_log=True)
    X = affine_transform(X0, is_log=True)
    X_test = affine_transform(X0_test, is_log=True)
else: 
    X_train = X0_train
    X = X0
    X_test = X0_test

# set bounds of the hyperparam opt and initial
# --------------------------------------------

# hyperparms for the buckling + GP kernel
kernel = buckling_RQ_kernel
# [relu_alph, gamma_coeff, RQ_coeff, length, alpha, constant, log10(sigma_n)]
lbounds = np.array([0.1] + [args.gammalb] + [1e-3] + [0.1] + [1e-4]*2 + [-4])
theta0 = np.array([5.0, 1.0, 1e-1, 1.0, 2.0, 1.0] + [-2])
ubounds = np.array([10.0]*6 + [0])

if not args.opt:
    # on old dataset: was pushing to high RQ coeff (bad extrapolation then)
    # axial_theta_opt = np.array([ 6.1113998 ,  0.93618111,  1.02880069,  2.6260183 ,  3.61487815, 0.5955076 , -3.79359076])

    # opt designs at current axial hyperparam optimization (for axial ntrain=2000)
    axial_theta_opt=np.array([4.99630718443544, 0.9999004325011723, 0.03583481403808614, 1.1524101376060423, 1.9959891162767658, 0.9999560129887785, -1.7679458065241216])
    axial_theta_opt=np.array([4.981801992541742, 0.9995130001726854, 0.02058467660235196, 0.9417197218475319, 1.9476959841602728, 0.9996652322809855, -1.7103460293491948])
    axial_theta_opt=np.array([4.967179290402711, 0.9990411895159346, 0.01930761236310141, 0.8104718784236444, 1.8790979147849334, 0.9992958917469411, -1.6308058379603843])
    axial_theta_opt=np.array([4.770866756384451, 0.988243783336739, 0.058580859840164944, 1.063360767037736, 0.136915375688455, 0.9897319273833416, -1.5298378328595506])
    axial_theta_opt=np.array([4.24924529696483, 0.9668667355603373, 0.23404459151290297, 6.603669025547548, 0.002219737106821394, 0.9638183801378273, -2.1444063661684476])
    # final axial theta opt
    # axial_theta_opt=np.array([1.49083009e+00, 9.43316328e-01, 3.57081802e-01, 1.00000000e+01, 9.90789055e-04, 9.58937173e-01, -2.24874104e+00])
    axial_theta_opt=np.array([ 1.87597327e+00,  6.87123481e-01,  5.73982474e-01,  8.86433206e+00,
        2.64332885e-03,  7.20043093e-01, -1.93544570e+00])

    theta0 = axial_theta_opt
    # theta0 = theta0


# prelim train the model for plotting
# -----------------------------------

x_train_L = tf.expand_dims(X_train, axis=1)
x_train_R = tf.expand_dims(X_train, axis=0)
sigma_n = np.power(10.0, theta0[-1]) 
nugget_term = sigma_n**2 * np.eye(x_train_L.shape[0])
K_train0 = kernel(x_train_L, x_train_R, theta0) + nugget_term
alpha_train0 = np.linalg.solve(K_train0, Y_train)

print(f"{sigma_n=}")

# plot the initial model
# ----------------------

plot_3d_gamma(
    X=X, Y=Y,
    X_train=X_train,
    kernel_func=kernel,
    theta=theta0,
    alpha_train=alpha_train0,
    #xi_bin=[0.3, 1.1],
    xi_bin=[0.6, 0.8],
    # xi_bin=[0.2, 0.6],
    zeta_bin=[0.0, 0.5],
    #zeta_bin=[0.5, 1.5],
    folder_name=f"output",
    load_name=args.load,
    file_prefix=f"init_{args.load}",
    is_affine=use_affine,
    show=args.show,
    save_npz=False, 
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
    folder_name=f"output",
    load_name=args.load,
    file_prefix=f"init_{args.load}",
    is_affine=use_affine,
    show=args.show,
    save_npz=False,
)

# predict error on test dataset initially
# ----------------------------------------
x_test_L = tf.expand_dims(X_test, axis=1)
K_test_train0 = kernel(x_test_L, x_train_R, theta0)
Y_test_pred = np.dot(K_test_train0, alpha_train0)

init_Rsq = eval_Rsquared(Y_test_pred, Y_test)
txt_hdl.write("--------------------------------------------\n\n")
txt_hdl.write(f"init Rsquared on ntrain={Y_train.shape[0]} n_test={Y_test.shape[0]} data\n")
txt_hdl.write(f"\t{init_Rsq=}:\n")
txt_hdl.write("--------------------------------------------\n\n")
txt_hdl.flush()


# optimize 
# --------

if args.opt:

    # now we need to do kfold hyperparameter optimization
    # ---------------------------------------------------

    folder_name = "output"
    theta_opt_dict = kfold_hyperparameter_optimization(
        kernel,
        lbounds=lbounds,
        theta0=theta0,
        ubounds=ubounds,
        num_kfolds=args.kfold,
        X=X_train, Y=Y_train,
        snopt_options={
            "Major iterations limit": 500, # debug change 2,
            "Print file": f"{folder_name}/opt/{base_name}_SNOPT_print.out",
            "Summary file": f"{folder_name}/opt/{base_name}_SNOPT_summary.out",
        },
        can_print=True,
    )

    theta_opt = np.array(theta_opt_dict['theta'])
    print(f"{theta_opt=}")

    txt_hdl.write("--------------------------------------------\n\n")
    txt_hdl.write(f"hyperparameter optimization with kfolds={args.kfold} and N={args.ntrain}\n")
    txt_hdl.write("\ttheta_opt:\n")
    txt_hdl.write(f"\t{theta_opt=}\n\n")
    txt_hdl.write("--------------------------------------------\n\n")


    # train the model again
    # ---------------------

    sigma_n = np.power(10.0, theta_opt[-1]) 
    nugget_term = sigma_n**2 * np.eye(x_train_L.shape[0])
    K_train_opt = kernel(x_train_L, x_train_R, theta_opt) + nugget_term
    alpha_train_opt = np.linalg.solve(K_train_opt, Y_train)

    # plot after hyperparam opt
    # -------------------------

    plot_3d_gamma(
        X=X, Y=Y,
        X_train=X_train,
        kernel_func=kernel,
        theta=theta_opt,
        alpha_train=alpha_train_opt,
        xi_bin=[0.6, 0.8],
        zeta_bin=[0.0, 0.5],
        folder_name=f"output",
        load_name=args.load,
        file_prefix=base_name,
        is_affine=use_affine,
        show=args.show,
    )

    plot_3d_xi(
        X=X, Y=Y,
        X_train=X_train,
        kernel_func=kernel,
        theta=theta_opt,
        alpha_train=alpha_train_opt,
        gamma_bin=[0.0, 0.1],
        zeta_bin=[0.0, 1.0],
        folder_name=f"output",
        load_name=args.load,
        file_prefix=base_name,
        is_affine=use_affine,
        show=args.show,
    )

    # predict error on test dataset after optimization
    # ----------------------------------------
    K_test_train_opt = kernel(x_test_L, x_train_R, theta_opt)
    Y_test_pred_opt = np.dot(K_test_train_opt, alpha_train_opt)

    opt_Rsq = eval_Rsquared(Y_test_pred_opt, Y_test)
    txt_hdl.write("--------------------------------------------\n\n")
    txt_hdl.write(f"opt Rsquared on ntrain={Y_train.shape[0]} n_test={Y_test.shape[0]} data\n")
    txt_hdl.write(f"\t{opt_Rsq=}:\n")
    txt_hdl.write("--------------------------------------------\n\n")

# close the text file
txt_hdl.close()
