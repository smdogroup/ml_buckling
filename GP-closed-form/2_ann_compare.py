# demo a squared exponential GP on the axial dataset, measure interp and extrap error
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import tensorflow as tf

sys.path.append("src/")
from closed_form_dataset import get_closed_form_data, split_data
from eval_GPs import eval_GPs, eval_ann
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
parser.add_argument("--epochs", type=int, default=None)
parser.add_argument("--dataf", type=float, default=1.0, help='fraction of full dataset size (out of 1000) for doing less trials')
parser.add_argument("--ks", type=float, default=None, help="shear_ks_param")
parser.add_argument("--activ", type=str, default='relu', help="relu or tanh")

parser.add_argument("--seed", type=int, default=1234, help="Seed for reproducibility (only used if --random is False)")
args = parser.parse_args()

# make the folder for this kernel
# -------------------------------
if not os.path.exists("output"):
    os.mkdir("output")

folder_name = f"output/ANN_{args.activ}"
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
base_name = f"tf_{ks_str}{axial_str}_{affine_str}_{log_str}"

# make a file in this folder for this specific case
txt_file = f"{folder_name}/{base_name}.txt"
txt_hdl = open(txt_file, "w")


# get interpolation dataset
# -------------------------

n_data_mult = args.dataf**(1.0/3.0)
n_rho = int(20 * n_data_mult)
n_gamma = int(10 * n_data_mult)
n_xi = np.max([int(5 * n_data_mult), 3])

n_data = n_rho * n_gamma * n_xi
print(f"{n_data_mult=} => {n_data=}")

X_interp, Y_interp = get_closed_form_data(
    axial=args.axial, 
    include_extrapolation=False, 
    affine_transform=args.affine, 
    log_transform=args.log,
    # n_rho0=20, n_gamma=10, n_xi=5,
    n_rho0=n_rho, n_gamma=n_gamma, n_xi=n_xi,
)

if args.epochs is None:
    epochs=400 if args.activ == 'tanh' else 100 # for relu
else:
    epochs = args.epochs

# # eval GPs metrics - R^2
# # ---------------------------
# interp_Rsq, extrap_Rsq = eval_ann(
#     n_dense=64,
#     epochs=epochs,
#     activation=args.activ,
#     # epochs=10,
#     # n_trials=30,
#     n_trials=5,
#     train_test_frac=0.8,
#     shear_ks_param=None,
#     axial=args.axial,
#     affine=args.affine,
#     log=args.log,
#     # n_rho0=20, n_gamma=10, n_xi=5,
#     n_rho0=n_rho, n_gamma=n_gamma, n_xi=n_xi,
#     metric_func=eval_Rsquared,
#     percentile=50.0,
# )

# # txt_hdl.write("-------------\n\n")
# txt_hdl.write(f"R^2 metrics:\n")
# txt_hdl.write(f"\t{interp_Rsq=}\n\n")
# txt_hdl.write(f"\t{extrap_Rsq=}\n\n")
# txt_hdl.write("--------------------------------------------\n\n")

# # eval GPs metrics - RMSE
# # ---------------------------
# interp_rmse, extrap_rmse = eval_ann(
#     n_dense=64,
#     epochs=epochs,
#     # n_trials=30,
#     n_trials=5,
#     activation=args.activ,
#     train_test_frac=0.8,
#     shear_ks_param=None,
#     axial=args.axial,
#     affine=args.affine,
#     log=args.log,
#     # n_rho0=20, n_gamma=10, n_xi=5,
#     n_rho0=n_rho, n_gamma=n_gamma, n_xi=n_xi,
#     metric_func=eval_rmse,
#     percentile=50.0,
# )

# # txt_hdl.write("-------------\n\n")
# txt_hdl.write(f"RMSE metrics:\n")
# txt_hdl.write(f"\t{interp_rmse=}\n\n")
# txt_hdl.write(f"\t{extrap_rmse=}\n\n")
# txt_hdl.write("--------------------------------------------\n\n")



# train the model for plotting
# ----------------------------

# now split into train and test for interpolation zone
X_train, Y_train, X_test, Y_test = split_data(X_interp, Y_interp, train_test_split=0.9)

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


import time
start_time = time.time()


# try making a tensorflow model here
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(3,)),
    tf.keras.layers.Dense(64, activation=args.activ),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
])

# predictions = model(X_train[:1]).numpy()
# print(f"{predictions.shape=}")

loss_fn = tf.keras.losses.MSE

# mse = loss_fn(Y_train[:1], predictions).numpy()
# print(f"{mse=}")



# compile and train tf model
model.compile(optimizer='adam',
            loss=loss_fn,
            metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=epochs)
model.evaluate(X_test,  Y_test, verbose=2)

# make predictionss on plot data
Y_plot_pred = model(X_plot).numpy()


train_time_sec = time.time() - start_time
print(f"{train_time_sec=:.4e}")

exit()

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


print(f"{n_data_mult=} => {n_data=}")

# cleanup
# -------

txt_hdl.close()



print(f"R^2 metrics:\n")
print(f"\t{interp_Rsq=}\n\n")
print(f"\t{extrap_Rsq=}\n\n")
print("--------------------------------------------\n\n")


print(f"{train_time_sec=:.4e}")