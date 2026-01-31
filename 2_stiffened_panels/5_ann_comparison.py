import sys
import pandas as pd
import numpy as np
import argparse
import tensorflow as tf
import os

sys.path.append("src/")
from kernel_library import *
# from plot_utils import plot_3d_gamma, plot_3d_xi
from plot_utils2 import plot_3d_gamma_tf, plot_3d_xi_tf
from data_transforms import affine_transform
from hyperparam_opt import kfold_hyperparameter_optimization
from eval_utils import *

# argparse
# --------

parent_parser = argparse.ArgumentParser(add_help=False)
parent_parser.add_argument("--load", type=str, default="Nx")
parent_parser.add_argument("--ntrain", type=int, default=1500) # 1000
parent_parser.add_argument("--gammalb", type=int, default=1e-1)
parent_parser.add_argument("--epochs", type=int, default=400)
parent_parser.add_argument("--opt", action=argparse.BooleanOptionalAction, default=True, help="Enable or disable axial mode (default: False)")
parent_parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=False, help="Show plot instead of save to file (default: False)")
parent_parser.add_argument("--archive", action=argparse.BooleanOptionalAction, default=False, help="Archive the hyperparams model (default: False)")
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
ntrain_str = f"ntrain{args.ntrain}"
gammalb_log10 = np.log10(args.gammalb)
gamma_lb_str = f"gammalb{args.gammalb}"
base_name = f"tf_{axial_str}_{ntrain_str}_{gamma_lb_str}"

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
    X_train = affine_transform(X0_train.copy(), is_log=True)
    X = affine_transform(X0, is_log=True)
    X_test = affine_transform(X0_test, is_log=True)
else: 
    X_train = X0_train
    X = X0
    X_test = X0_test

# print(f"{X0_train[0,:]=} {X_train[0,:]=}")
# exit()

# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0
# print(f"{x_train.shape=}")


print(f"{X_train.shape=} {X_test.shape=}")

ndense = 128
# ndense = 64

# try making a tensorflow model here
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(4,)),
  tf.keras.layers.Dense(ndense, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(1)
])

# predictions = model(X_train[:1]).numpy()
# print(f"{predictions.shape=}")

loss_fn = tf.keras.losses.MSE

# mse = loss_fn(Y_train[:1], predictions).numpy()
# print(f"{mse=}")

import time
start_time = time.time()

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=args.epochs)
model.evaluate(X_test,  Y_test, verbose=2)

# exit()

dt_sec = time.time() - start_time 
dt_hrs = dt_sec / 3600.0

print(f"ANN training time {dt_hrs=:.4e}")


# plot the initial model
# ----------------------

thick_walled = False
if thick_walled:
    xi_bin = [0.4, 0.8]
    zeta_bin = [0.5, 1.5]
else:
    xi_bin = [0.6, 0.8]
    zeta_bin = [0.0, 0.5]

#xi_bin = [0.0, 0.0]
#zeta_bin = [0.0, 0.0]

plot_3d_gamma_tf(
    X=X, Y=Y,
    X_train=X_train,
    tf_model=model,
    theta=None,
    alpha_train=None,
    xi_bin=xi_bin,
    zeta_bin=zeta_bin,
    folder_name=f"output",
    load_name=args.load,
    file_prefix=f"tf_{args.load}",
    is_affine=use_affine,
    show=args.show,
    save_npz=False, 
)

#print(f'{X0_train.shape} {X.shape=}')
# exit()

plot_3d_xi_tf(
    X=X, Y=Y,
    X_train=X_train,
    tf_model=model,
    theta=None,
    alpha_train=None,
    gamma_bin=[0.0, 0.1],
    zeta_bin=[0.0, 1.0],
    folder_name=f"output",
    load_name=args.load,
    file_prefix=f"tf_{args.load}",
    is_affine=use_affine,
    show=args.show,
    save_npz=False,
)

# predict error on test dataset initially
# ----------------------------------------
Y_test_pred = model(X_test).numpy()

print(f"{Y_test_pred.shape=} {Y_test.shape=}")

import matplotlib.pyplot as plt
plt.plot(Y_test, Y_test_pred)
plt.show()

init_Rsq = eval_Rsquared(Y_test_pred, Y_test)
txt_hdl.write("--------------------------------------------\n\n")
txt_hdl.write(f"init Rsquared on ntrain={Y_train.shape[0]} n_test={Y_test.shape[0]} data\n")
txt_hdl.write(f"\t{init_Rsq=}:\n")
txt_hdl.write("--------------------------------------------\n\n")
txt_hdl.flush()
print(f"{init_Rsq=}")


# # optimize 
# # --------

# if args.opt:

#     # now we need to do kfold hyperparameter optimization
#     # ---------------------------------------------------

#     folder_name = "output"
#     theta_opt_dict = kfold_hyperparameter_optimization(
#         kernel,
#         lbounds=lbounds,
#         theta0=theta0,
#         ubounds=ubounds,
#         num_kfolds=args.kfold,
#         X=X_train, Y=Y_train,
#         snopt_options={
#             "Major iterations limit": 500, # debug change 2,
#             "Print file": f"{folder_name}/opt/{base_name}_SNOPT_print.out",
#             "Summary file": f"{folder_name}/opt/{base_name}_SNOPT_summary.out",
#         },
#         can_print=True,
#     )

#     theta_opt = np.array(theta_opt_dict['theta'])
#     print(f"{theta_opt=}")

#     txt_hdl.write("--------------------------------------------\n\n")
#     txt_hdl.write(f"hyperparameter optimization with kfolds={args.kfold} and N={args.ntrain}\n")
#     txt_hdl.write("\ttheta_opt:\n")
#     txt_hdl.write(f"\t{theta_opt=}\n\n")
#     txt_hdl.write("--------------------------------------------\n\n")


#     # train the model again
#     # ---------------------

#     sigma_n = np.power(10.0, theta_opt[-1]) 
#     nugget_term = sigma_n**2 * np.eye(x_train_L.shape[0])
#     K_train_opt = kernel(x_train_L, x_train_R, theta_opt) + nugget_term
#     alpha_train_opt = np.linalg.solve(K_train_opt, Y_train)

#     # plot after hyperparam opt
#     # -------------------------

#     plot_3d_gamma(
#         X=X, Y=Y,
#         X_train=X_train,
#         kernel_func=kernel,
#         theta=theta_opt,
#         alpha_train=alpha_train_opt,
#         xi_bin=[0.6, 0.8],
#         zeta_bin=[0.0, 0.5],
#         folder_name=f"output",
#         load_name=args.load,
#         file_prefix=base_name,
#         is_affine=use_affine,
#         show=args.show,
#     )

#     plot_3d_xi(
#         X=X, Y=Y,
#         X_train=X_train,
#         kernel_func=kernel,
#         theta=theta_opt,
#         alpha_train=alpha_train_opt,
#         gamma_bin=[0.0, 0.1],
#         zeta_bin=[0.0, 1.0],
#         folder_name=f"output",
#         load_name=args.load,
#         file_prefix=base_name,
#         is_affine=use_affine,
#         show=args.show,
#     )

#     # predict error on test dataset after optimization
#     # ----------------------------------------
#     K_test_train_opt = kernel(x_test_L, x_train_R, theta_opt)
#     Y_test_pred_opt = np.dot(K_test_train_opt, alpha_train_opt)

#     opt_Rsq = eval_Rsquared(Y_test_pred_opt, Y_test)
#     txt_hdl.write("--------------------------------------------\n\n")
#     txt_hdl.write(f"opt Rsquared on ntrain={Y_train.shape[0]} n_test={Y_test.shape[0]} data\n")
#     txt_hdl.write(f"\t{opt_Rsq=}:\n")
#     txt_hdl.write("--------------------------------------------\n\n")

# # close the text file
# txt_hdl.close()

# if args.archive:
#     import ml_buckling as mlb

#     # archive the data to the format of the
#     filename = "axialGP.csv" if args.load == "Nx" else "shearGP.csv"
#     output_csv = "../archived_models/" + filename

#     # remove the previous csv file if it exits
#     # assume on serial here
#     if os.path.exists(output_csv):
#         os.remove(output_csv)

#     # print(f"{X_train=}")

#     if args.opt:
#         alpha = alpha_train_opt
#     else:
#         alpha = alpha_train0

#     # [log(1+xi), log(rho0), log(1+gamma), log(1+10^3 * zeta)]
#     dataframe_dict = {
#         # order was flipped from initial order
#         "log(1+xi)": X0_train[:, 1],
#         "log(rho0)": X0_train[:, 0], # before affine transform
#         "log(1+gamma)": X0_train[:, 2],
#         "log(1+10^3*zeta)": X0_train[
#             :, 3
#         ],  # gamma,zeta are flipped to the order used in TACS
#         "alpha": alpha[:, 0],
#     }
#     model_df = pd.DataFrame(dataframe_dict)
#     model_df.to_csv(output_csv)

#     # also deploy the current theta_opt
#     theta_csv = mlb.axial_theta_csv if args.load == "Nx" else mlb.shear_theta_csv
#     if os.path.exists(theta_csv):
#         os.remove(theta_csv)
    
#     # print(f"{theta_csv=}")
#     if args.opt:
#         theta = theta_opt
#     else:
#         theta = theta0

#     # exclude the noise though

#     theta_df_dict = {
#         "theta": theta[:-1],
#     }
#     theta_df = pd.DataFrame(theta_df_dict)
#     theta_df.to_csv(theta_csv)
