import sys
import numpy as np
import pandas as pd
import os
from pyoptsparse import SNOPT, Optimization

sys.path.append("src/")
from closed_form_dataset import get_closed_form_data, split_data
from eval_GPs import eval_GPs
from kernel_library import *

kernel = SE_kernel

ntheta = 3
lbounds = np.array([0.1]*3)
theta0 = np.array([1.0, 8.0, 4.0])
ubounds = np.array([10.0]*3)

def nMAP(theta):
    X, Y = get_closed_form_data(
        axial=True, include_extrapolation=False, 
        affine_transform=True, 
        log_transform=True,
        n_rho0=20, n_gamma=10, n_xi=5,
    )

    nMAP_sum = 0.0
    for itrial in range(30):
        X_train, Y_train, X_test, Y_test = \
            split_data(X, Y, train_test_split=0.9)
        
        x_train_L = tf.expand_dims(X_train, axis=1)
        x_train_R = tf.expand_dims(X_train, axis=0)

        sigma_n = 1e-2
        nugget = sigma_n**2 * np.eye(x_train_L.shape[0])
        K_train = kernel(x_train_L, x_train_R, theta) + nugget

        alpha = np.linalg.solve(K_train, Y_train)
        term1 = (0.5 * np.dot(Y_train, alpha) )
        term2 = (0.5 * np.abs(np.linalg.slogdet(K_train)))[1]
        # term3 = n_train / 2.0 * np.log(2.0 * np.pi)
        # return term1 + term2
        nMAP_sum += (term1 + term2)

    avg_nMAP = nMAP_sum / 30

    # add regularization term for SE as if theta are 1/lengths
    lam = 500.0
    reg_term = lam * np.sum(np.log(1.0/np.array(theta)))
    print(f'{reg_term=}')
    obj = avg_nMAP + reg_term

    return obj, None

def extrap_RMSE(theta):
    X, Y = get_closed_form_data(
        axial=True, include_extrapolation=False, 
        affine_transform=True, 
        log_transform=True,
        n_rho0=20, n_gamma=10, n_xi=5,
    )

    avg_extrap_RMSE = 0.0

    for itrial in range(30):
        X_train, Y_train, X_test, Y_test = \
            split_data(X, Y, train_test_split=0.9)

        X_extrap, Y_extrap_truth = get_closed_form_data(
            axial=True, 
            include_extrapolation=True,
            include_interpolation=False, 
            affine_transform=True, 
            shear_ks_param=1.0,
            log_transform=True,
            # no need for nan_interp, nan_extrap inputs yet that's for plotting
            n_rho0=20, n_gamma=10, n_xi=5,
        )

        x_train_L = tf.expand_dims(X_train, axis=1)
        x_train_R = tf.expand_dims(X_train, axis=0)

        sigma_n = 1e-2
        nugget = sigma_n**2 * np.eye(x_train_L.shape[0])
        K_train = kernel(x_train_L, x_train_R, theta) + nugget

        alpha_train = np.linalg.solve(K_train, Y_train)

        x_extrap_L = tf.expand_dims(X_extrap, axis=1)
        K_cross_extrap = kernel(x_extrap_L, x_train_R, theta)
        Y_extrap_pred = np.dot(K_cross_extrap, alpha_train)
        sq_diff_extrap = (Y_extrap_truth - Y_extrap_pred)**2
        test_extrap_RMSE = np.sqrt(np.mean(sq_diff_extrap))
        avg_extrap_RMSE += test_extrap_RMSE
    avg_extrap_RMSE /= 30.0

    return avg_extrap_RMSE, None
    

def nMAP_pyos(theta_dict):
    theta = list(theta_dict["theta"])
    func_val, alpha = nMAP(theta)
    # func_val, alpha = extrap_RMSE(theta)

    # avg_rel_err = post_evaluate(theta, alpha)
    funcs = {"obj": func_val}
    # if comm.rank == 0:
    # print(f"{func_val=}, {avg_rel_err=}, {theta=}")
    print(f"{func_val=}, {theta=} {func_val=}")
    return funcs, False  # fail = False

# use pyoptsparse to setup the optimization problem
# Optimization Object
optProb = Optimization("nMap hyperparameter optimization", nMAP_pyos)

# Design Variables
optProb.addVarGroup("theta", ntheta, lower=lbounds, value=theta0, upper=ubounds)
optProb.addObj("obj")

# Optimizer
snoptimizer = SNOPT({})
sol = snoptimizer(
    optProb,
    # sens=nMAP_grad_pyos,
    # storeHistory="nmap.hst",
    # hotStart="nmap.hst",
)

sol_xdict = sol.xStar
print(f"Final solution = {sol_xdict}", flush=True)
