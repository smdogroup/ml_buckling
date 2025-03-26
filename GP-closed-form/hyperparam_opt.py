import sys
import numpy as np
import pandas as pd
import os
from pyoptsparse import SNOPT, Optimization

sys.path.append("src/")
from closed_form_dataset import get_closed_form_data, split_data
from eval_GPs import eval_GPs
from kernel_library import *

# inputs
# -----------------------

# kernel_num = 1
# kernel_num = 5
kernel_num = 6

# increasing num k folds improved the rational quadratic kernel..
# num_k_folds = 5
# num_k_folds = 10 # try to increase it
num_k_folds = 20


# -----------------------

if kernel_num == 1:
    kernel = SE_kernel
    ntheta = 3
    lbounds = np.array([0.1]*3)
    theta0 = np.array([1.0, 8.0, 4.0])
    ubounds = np.array([10.0]*3)

elif kernel_num == 5:
    kernel = custom_kernel1
    ntheta = 7
    relu_alph = 5.0 # fixed value (so that smoothness is fixed, otherwise makes not smooth)
    # try adding relu_alph back into optimization.. before was fixed
    # had lower bound of th[5] set to 1e-2 for a bit, may not need that now

    # need lower bound for SE term to get smaller so better extrap!
    lbounds = np.array([0.1]*3 + [1e-2] + [0.1, 1e-3, 0.1])
    ubounds = np.array([10.0]*3 + [10.0] + [10.0]*3)
    theta0 = np.array([1.0, 8.0, 4.0, relu_alph, 1.0, 0.1, 0.1])

elif kernel_num == 6:

    kernel = custom_kernel2 

    ntheta = 6
    lbounds = np.array([0.1]*2 + [1e-3] + [0.1]*3)
    ubounds = np.array([10.0]*ntheta)
    theta0 = np.array([5.0, 1.0, 1e-1, 1.0, 2.0, 1.0])
    # theta means [relu_alph, gamma_coeff, RQ_coeff, RQ_length, RQ_alpha, constant]


# use k-fold cross validation this time, with MSE loss
X, Y = get_closed_form_data(
    axial=True, include_extrapolation=False, 
    affine_transform=True, 
    log_transform=True,
    n_rho0=20, n_gamma=10, n_xi=5,
)

# split into k = 5 distinct sets of data, can increase k maybe too later
k = num_k_folds
Xlist = []
Ylist = []
N = X.shape[0]
# print(f"{N}")
n_part = N // k
for i_k in range(k):
    Xlist += [X[n_part*i_k:n_part*(i_k+1), :]]
    Ylist += [Y[n_part*i_k:n_part*(i_k+1)]]
    # print(f"{Xlist[i_k].shape[0]}")
# exit()


def mse_loss(theta):
    # k-fold mse loss useful in estimating generalization error of the model

    my_mse = 0.0

    # loop over k-fold datasets
    for i_k in range(k):
        Xtrain = np.concatenate([Xlist[i] for i in range(k) if not(i == i_k)], axis=0)
        Ytrain = np.concatenate([Ylist[i] for i in range(k) if not(i == i_k)], axis=0)
        Xtest = Xlist[i_k]
        Ytest = Ylist[i_k]
        # print(f"{Xtrain.shape[0]}")
        # print(f"{Xtest.shape[0]}")

        x_train_L = tf.expand_dims(Xtrain, axis=1)
        x_train_R = tf.expand_dims(Xtrain, axis=0)

        sigma_n = 1e-2
        nugget = sigma_n**2 * np.eye(x_train_L.shape[0])
        K_train = kernel(x_train_L, x_train_R, theta) + nugget
        alpha = np.linalg.solve(K_train, Ytrain)

        x_test_L = tf.expand_dims(Xtest, axis=1)
        K_test_train = kernel(x_test_L, x_train_R, theta)
        Y_test_pred = np.dot(K_test_train, alpha)

        # mse loss now
        Y_resid = Ytest - Y_test_pred
        my_mse += np.mean(Y_resid**2)

    # average mse loss
    my_mse /= k
    # print(F"{my_mse=}")
    return my_mse

# mse_loss(theta0)

def obj(theta_dict):
    theta = list(theta_dict["theta"])
    func_val = mse_loss(theta)
    # func_val, alpha = extrap_RMSE(theta)

    # avg_rel_err = post_evaluate(theta, alpha)
    funcs = {"obj": func_val}
    # if comm.rank == 0:
    # print(f"{func_val=}, {avg_rel_err=}, {theta=}")
    print(f"{func_val=}, {theta=} {func_val=}")
    return funcs, False  # fail = False

# use pyoptsparse to setup the optimization problem
# Optimization Object
optProb = Optimization("nMap hyperparameter optimization", obj)

# Design Variables
optProb.addVarGroup("theta", ntheta, lower=lbounds, value=theta0, upper=ubounds)
optProb.addObj("obj", scale=1e2)

# Optimizer
snoptimizer = SNOPT({
    'Major Optimality tol' : 1e-6, # can increase potentially
})
sol = snoptimizer(
    optProb,
    # sens=nMAP_grad_pyos,
    # storeHistory="nmap.hst",
    # hotStart="nmap.hst",
)

sol_xdict = sol.xStar
print(f"Final solution = {sol_xdict}", flush=True)
