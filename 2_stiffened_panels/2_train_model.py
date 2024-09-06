import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import niceplots, scipy, time, os
import argparse
from mpl_toolkits import mplot3d
from matplotlib import cm
import shutil, random
from pyoptsparse import SNOPT, Optimization
from mpi4py import MPI

# parallel matrix solving in OpenBLAS from np.linalg.solve
from os import environ
environ['OMP_NUM_THREADS'] = '4'
from _saved_kernel import kernel, soft_abs, soft_relu

"""
@Author : Sean Engelstad
Train the model using hyperparameter optimization for -log p(y|X,theta) objective function
"""

comm = MPI.COMM_WORLD

np.random.seed(1234567)

# parse the arguments
parent_parser = argparse.ArgumentParser(add_help=False)
parent_parser.add_argument("--load", type=str, default="Nx")
parent_parser.add_argument("--ntrain", type=int, default=3000)
parent_parser.add_argument("--ntest", type=int, default=1000)
parent_parser.add_argument('--debug1', default=False, action=argparse.BooleanOptionalAction)
parent_parser.add_argument('--checkderivs', default=False, action=argparse.BooleanOptionalAction)
parent_parser.add_argument('--checkderivs2', default=False, action=argparse.BooleanOptionalAction)

args = parent_parser.parse_args()

assert args.load in ["Nx", "Nxy"]

load = args.load

# load the Nxcrit dataset
csv_filename = f"{load}_stiffened"
df = pd.read_csv("data/" + csv_filename + ".csv")

# extract only the model columns
X = df[["log(1+xi)", "log(rho_0)", "log(1+10^3*zeta)", "log(1+gamma)"]].to_numpy()
Y = df["log(eig_FEA)"].to_numpy()
Y = np.reshape(Y, newshape=(Y.shape[0], 1))

N_data = X.shape[0]

n_train = args.ntrain
n_test = args.ntest

# kernel_option == 10

# each sublist is [lb, value, ub]
variables = [
    [-0.5, 0.0, 0.5],  # 0 - gamma_rho_dist gamma-slope
    [0.02, 0.1, 0.3],  # 1 - BL kernel constant
    [0.1, 1.0, 5.0],   # 2 - BL kernel slope
    [0.01, 0.1, 3.0],  # 3 - gamma kernel slope
    [0.01, 0.1, 3.0],  # 4 - xi kernel slope
    [1e-3, 0.01, 0.1], # 5 - xi kernel quadratic slope
    [1e-3, 0.02, 0.1], # 6 - zeta kernel slope
    [1e-3, 0.01, 0.1], # 7 - zeta kernel quad slope
    [0.01, 0.05, 1.0], # 8 - SE kernel leading coeff
    [0.05, 0.2, 1.0],  # 9 - SE kernel rho0* length scale
    [5.0, 6.0, 50.0],   # 10 - SE kernel gamma length scale (want to be pretty long length scale otherwise buckling loads return to zero in middle)
    [0.5, 1.0, 2.0],   # 11 - window kernel => window length
    [1e-4, 1e-1, 1e0], # 12 - sigma_n noise
]

ntheta = len(variables)
lbounds = np.array([var[0] for var in variables])
values = np.array([var[1] for var in variables])
ubounds = np.array([var[2] for var in variables])
theta0 = values

def kernel_deriv(xp, xq, theta, idx, include_sigma=True):
    """compute derivative of kernel function above w.r.t. the dv index"""
    # use complex-step method here as it is very convenient
    dtheta = np.zeros((ntheta,))
    dtheta[idx] = 1e-30
    f1 = kernel(xp, xq, theta + dtheta * 1j, include_sigma=True)
    return np.imag(f1) / 1e-30

if comm.rank == 0:
    print(f"Monte Carlo #data training {n_train} / {X.shape[0]} data points")

# randomly permute the arrays
rand_perm = np.random.permutation(N_data)
X = X[rand_perm, :]
Y = Y[rand_perm, :]

# if comm.rank == 0:
#     Y_max = np.max(np.abs(Y))
#     print(f"{Y_max=}")
#     Y_min = np.min(np.abs(Y))
#     print(f"{Y_min=}")
# exit()

X_train = X[:n_train, :]
Y_train = Y[:n_train, :]
X_test = X[n_train:(n_train+n_test), :]
Y_test = Y[n_train:(n_train+n_test), :]
n_test = X_test.shape[0] # overwrite if n_train + n_test > N_data

def nMAP(theta):
    """
    the negative max a posteriori objective function (to minimize)
    eqn is:
    obj = -log p(y|X,theta) = 1/2 * y^T * K^-1 * y + 1/2 * log det K + N/2 * log(2*pi)
        where K = Cov(Xtrain,Xtrain;theta) 
    You can use the Cholesky decomp as an efficient way to compute obj but then the differentiation is a bit harder.
    """
    K_y = np.array(
        [
            [kernel(X_train[i, :], X_train[j, :], theta) for i in range(n_train)]
            for j in range(n_train)
        ]
    ) + theta[ntheta-1]**2 * np.eye(n_train)

    alpha = np.linalg.solve(K_y, Y_train)
    term1 = (0.5 * Y_train.T @ alpha)[0,0]
    term2 = (0.5 * np.abs(np.linalg.slogdet(K_y)) )[1]
    term3 = n_train/2.0 * np.log(2.0 * np.pi)
    # print(f"{term1=}")
    # print(f"{term2=}")
    # print(f"{term3=}")

    return term1 + term2 + term3, alpha

def nMAP_grad(theta, can_print=False):
    """
    gradient of the the negative max a posteriori objective function (to minimize)
    
    our obj = - 1/2 * log p(y|X,theta)
    our grad_j = dobj/dtheta_j

    grad_j = -1/2 * tr((alpha * alpha.T - K^-1) * dK/dtheta_j)
        where alpha = K^-1 * Y
    """
    K_y = np.array(
        [
            [kernel(X_train[i, :], X_train[j, :], theta) for i in range(n_train)]
            for j in range(n_train)
        ]
    ) + theta[ntheta-1]**2 * np.eye(n_train)

    # forward part (not gradient yet)
    alpha = np.linalg.solve(K_y, Y_train)
    term1 = (0.5 * Y_train.T @ alpha)[0,0]
    term2 = (0.5 * np.abs(np.linalg.slogdet(K_y)) )[1]
    term3 = n_train/2.0 * np.log(2.0 * np.pi)
    nmap = term1 + term2 + term3

    # now get the gradient..
    grad = np.zeros((ntheta,))
    for itheta in range(ntheta):
        if can_print: print(f"\tgetting grad entry theta{itheta}")
        # get dK/dtheta_j
        if itheta < (ntheta-1):
            Kgrad = np.array(
                [
                    [kernel_deriv(X_train[i, :], X_train[j, :], theta, itheta) for i in range(n_train)]
                    for j in range(n_train)
                ]
            )
        elif itheta == (ntheta-1):
            Kgrad = 2 * theta[ntheta-1] * np.eye(n_train)

        # compute inner matrix terms
        term1_inner = alpha @ alpha.T @ Kgrad
        term2_inner = -1.0 * np.linalg.solve(K_y, Kgrad)
        inner_matrix = term1_inner - term2_inner

        deriv = -0.5 * np.trace(inner_matrix)
        grad[itheta] = deriv
    return grad

if args.debug1:
    import time
    start_time = time.time()
    print(f"starting nmap computation..")
    init_nmap = nMAP(theta0)
    end_time = time.time()
    dt = end_time - start_time
    print(f"\tdone with nmap in {dt} sec")
    print(f"\tinit nmap = {init_nmap}", flush=True)

    print("\bstarting nmap grad..")
    nmap_grad = nMAP_grad(theta0, can_print=True)
    end_time2 = time.time()
    dt2 = end_time2 - end_time
    print(f"\tdone with nmap grad in {dt2} sec")
    print(f"\tnmap grad = {nmap_grad}")
    exit()

# check the derivatives of the objective function using complex-step
if args.checkderivs:
    print("\nChecking nMAP gradient with complex-step method..")
    # use complex-step method to check the derivatives
    dtheta = np.random.rand(ntheta)
    f1,_ = nMAP(theta0 + dtheta * 1e-30 * 1j)
    cs_deriv = np.imag(f1) / 1e-30

    # use the exact gradient also
    grad = nMAP_grad(theta0, can_print=True)
    an_deriv = np.dot(grad, dtheta)

    # rel error and report findings
    rel_err = abs((cs_deriv - an_deriv) / an_deriv)
    print(f"{cs_deriv=}")
    print(f"{an_deriv=}")
    print(f"{rel_err=}")
    exit()

if args.checkderivs2:
    print("\nChecking nMAP gradient components with complex-step method..")
    # use complex-step method to check the derivatives
    dtheta = np.zeros((11,))
    idx = 0
    dtheta[idx] = 1.0
    f1,_ = nMAP(theta0 + dtheta * 1e-30 * 1j)
    cs_deriv = np.imag(f1) / 1e-30

    # use the exact gradient also
    grad = nMAP_grad(theta0, can_print=True)
    an_deriv = np.dot(grad, dtheta)

    # rel error and report findings
    rel_err = abs((cs_deriv - an_deriv) / an_deriv)
    print(f"{cs_deriv=}")
    print(f"{an_deriv=}")
    print(f"{rel_err=}")
    exit()

# define the post-evaluate method
def post_evaluate(theta, alpha):
    # predict and report the relative error on the test dataset
    K_test_cross = np.array(
        [
            [
                kernel(X_train[i, :], X_test[j, :], theta)
                for i in range(n_train)
            ]
            for j in range(n_test)
        ]
    )
    Y_test_pred = K_test_cross @ alpha

    crit_loads = np.exp(Y_test)
    crit_loads_pred = np.exp(Y_test_pred)

    abs_err = crit_loads_pred - crit_loads
    rel_err = abs(abs_err / crit_loads)
    avg_rel_err = np.mean(rel_err)
    return avg_rel_err

# TRAIN THE MODEL WITH HYPERPARAMETER OPTIMIZATION
# ------------------------------------------------

# redefine objective, gradient for pyoptsparse dict format
def nMAP_pyos(theta_dict):
    theta = list(theta_dict["theta"])
    func_val,alpha = nMAP(theta)
    avg_rel_err = post_evaluate(theta,alpha)
    funcs = {"obj" : func_val}
    if comm.rank == 0:
        print(f"{func_val=}, {avg_rel_err=}, {theta=}")
    return funcs, False # fail = False

def nMAP_grad_pyos(theta_dict, funcs):
    grad = nMAP_grad(theta_dict["theta"])
    funcs_sens = { "obj" : {"theta" : grad}}
    return funcs_sens, False # fail = False

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

# EVALUATE THE MODEL
# ------------------

# TODO
