import numpy as np
import tensorflow as tf
from closed_form_dataset import get_closed_form_data, split_data

def eval_GPs(
    kernel,
    theta,
    n_trials:int=30,
    sigma_n:float=1e-3,
    rand_seed:int=None, # None means random
    train_test_frac:float=0.8,
    shear_ks_param:float=None,
    axial:bool=True,
    affine:bool=True,
    log:bool=True,
    percentile:float=90.0, # out of 100.0
    n_rho0:int=20,
    n_gamma:int=10,
    n_xi:int=5,
    can_print:bool=False,
):

    np.random.seed(rand_seed)

    # get only interp data
    X, Y = get_closed_form_data(
        axial=axial, include_extrapolation=False, 
        affine_transform=affine, 
        log_transform=log,
        n_rho0=n_rho0, n_gamma=n_gamma, n_xi=n_xi,
    )

    # compute RMSE for some number of trials
    interp_RMSEs = []
    extrap_RMSEs = []
    for trial in range(n_trials):

        # now split into train and test for interpolation zone
        X_train, Y_train, X_test, Y_test = split_data(X, Y, train_test_split=train_test_frac)

        # fast way to compute kernel functions, from ml_pde_buckling final proj
        x_train_L = tf.expand_dims(X_train, axis=1)
        x_train_R = tf.expand_dims(X_train, axis=0)

        nugget = sigma_n**2 * np.eye(x_train_L.shape[0])
        K_train = kernel(x_train_L, x_train_R, theta) + nugget
        # print(f"{K_train.shape=}")
        # plt.imshow(K_train)

        # get the training weights
        alpha_train = np.linalg.solve(K_train, Y_train)

        # compute the interpolation error
        # ------------------------------

        x_test_L = tf.expand_dims(X_test, axis=1)
        K_cross_test = kernel(x_test_L, x_train_R, theta)
        Y_test_pred = np.dot(K_cross_test, alpha_train)
        sq_diff = (Y_test - Y_test_pred)**2
        test_interp_RMSE = np.sqrt(np.mean(sq_diff))
        # print(f"{test_interp_RMSE=}")

        # compute the extrapolation error
        # -------------------------------

        # get the extrapolation dataset
        X_extrap, Y_extrap_truth = get_closed_form_data(
            axial=axial, 
            include_extrapolation=True,
            include_interpolation=False, 
            affine_transform=affine, 
            shear_ks_param=shear_ks_param,
            log_transform=log,
            # no need for nan_interp, nan_extrap inputs yet that's for plotting
            n_rho0=n_rho0, n_gamma=n_gamma, n_xi=n_xi,
        )

        x_extrap_L = tf.expand_dims(X_extrap, axis=1)
        K_cross_extrap = kernel(x_extrap_L, x_train_R, theta)
        Y_extrap_pred = np.dot(K_cross_extrap, alpha_train)
        sq_diff_extrap = (Y_extrap_truth - Y_extrap_pred)**2
        test_extrap_RMSE = np.sqrt(np.mean(sq_diff_extrap))
        if can_print:
            print(f"{trial+1}/{n_trials}: {test_interp_RMSE=} {test_extrap_RMSE=}")
        interp_RMSEs += [test_interp_RMSE]
        extrap_RMSEs += [test_extrap_RMSE]

    # compute 90% RMSEs for each
    ovr_interp_RMSE = np.percentile(np.array(interp_RMSEs), q=percentile)
    ovr_extrap_RMSE = np.percentile(np.array(extrap_RMSEs), q=percentile)

    return ovr_interp_RMSE, ovr_extrap_RMSE