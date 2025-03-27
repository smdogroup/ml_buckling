from pyoptsparse import SNOPT, Optimization
import numpy as np
import tensorflow as tf

def kfold_hyperparameter_optimization(
    kernel_func,
    lbounds:np.ndarray,
    theta0:np.ndarray,
    ubounds:np.ndarray,
    num_kfolds:int=10,
    X=None,
    Y=None,
    snopt_options:dict={},
    can_print:bool=False,
):
    
    # split the data into k folds
    k = num_kfolds
    Xlist = []
    Ylist = []
    N = X.shape[0]
    # print(f"{N}")
    n_part = N // k

    # randomly shuffle the data
    Xshuffle = np.random.shuffle(X.copy())
    Yshuffle = np.random.shuffle(Y.copy())

    for i_k in range(k):
        Xlist += [Xshuffle[n_part*i_k:n_part*(i_k+1), :]]
        Ylist += [Yshuffle[n_part*i_k:n_part*(i_k+1)]]

    # define the mse loss
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
            K_train = kernel_func(x_train_L, x_train_R, theta) + nugget
            alpha = np.linalg.solve(K_train, Ytrain)

            x_test_L = tf.expand_dims(Xtest, axis=1)
            K_test_train = kernel_func(x_test_L, x_train_R, theta)
            Y_test_pred = np.dot(K_test_train, alpha)

            # mse loss now
            Y_resid = Ytest - Y_test_pred
            my_mse += np.mean(Y_resid**2)

        # average mse loss
        my_mse /= k
        # print(F"{my_mse=}")
        return my_mse
    
    def pyoptsparse_obj(theta_dict):
        theta = list(theta_dict["theta"])
        func_val = mse_loss(theta)

        funcs = {"obj": func_val}
        if can_print: print(f"{theta=} {func_val=}")
        return funcs, False  # fail = False
    
    # use pyoptsparse to setup the optimization problem
    # Optimization Object
    optProb = Optimization("nMap hyperparameter optimization", pyoptsparse_obj)

    # Design Variables
    ntheta = theta0.shape[0]
    optProb.addVarGroup("theta", ntheta, lower=lbounds, value=theta0, upper=ubounds)
    optProb.addObj("obj", scale=1e2)

    # Optimizer
    # print(f"{snopt_options=}")
    snoptimizer = SNOPT(options=snopt_options)
    sol = snoptimizer(optProb)

    sol_xdict = sol.xStar
    print(f"Final solution = {sol_xdict}", flush=True)

    return sol_xdict