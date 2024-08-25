import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import niceplots, scipy, time, os
import argparse
from mpl_toolkits import mplot3d
from matplotlib import cm
import shutil, random
from pyoptsparse import SNOPT, Optimization

"""
@Author : Sean Engelstad
Train the model using hyperparameter optimization for -log p(y|X,theta) objective function
"""

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
load_prefix = "Nx_stiffened" if load == "Nx" else "Nxy"
csv_filename = f"{load}_stiffened2" if load == "Nx" else f"{load}_stiffened"
df = pd.read_csv("data/" + csv_filename + ".csv")

# extract only the model columns
X = df[["log(1+xi)", "log(rho_0)", "log(1+10^3*zeta)", "log(1+gamma)"]].to_numpy()
Y = df["log(lam_star)"].to_numpy()
Y = np.reshape(Y, newshape=(Y.shape[0], 1))

N_data = X.shape[0]

n_train = args.ntrain
n_test = args.ntest


def relu(x):
    return max([0.0, x])

def soft_relu(x, rho=10):
    return 1.0 / rho * np.log(1 + np.exp(rho * x))

def soft_abs(x, rho=10):
    return 1.0 / rho * np.log(np.exp(rho*x) + np.exp(-rho*x))

# kernel_option == 10

theta0 = np.array([
    0.1, # BL_kernel constant
    0.05, # SE kernel factor
    0.2, # length scale for rho0 direction
    0.3, # length scale for gamma direction
    1.0, # boundary of window kernel rho0 direction
    0.3, # boundary of gamma SE window kernel
    1.0, # gamma kernel constant
    0.1, # gamma kernel slope
    0.1, # xi kernel slope
    0.02, # zeta kernel slope
    1e-1, # sigma_n noise
])

# opt_soln1 = OrderedDict([('theta', array([0.29999694, 0.96247034, 0.44112862, 0.23071223, 1.16697047,
#        0.69797125, 2.79486572, 0.8514289 , 0.9999892 , 0.01616109,
#        0.89938449]))])

bounds = [
    [0.02, 0.3], # BL_kernel constant
    [0.01, 1.0], # SE kernel factor
    [0.05, 1.0], # length scale for rho0 direction
    [0.05, 1.0], # length scale for gamma direction
    [0.5, 2.0], # boundary of window kernel rho0 direction
    [0.1, 1.0], # boundary of gamma SE window kernel
    [0.01, 5.0], # gamma kernel constant
    [0.01, 3.0], # gamma kernel slope
    [0.01, 3.0], # xi kernel slope
    [1e-3, 0.1], # zeta kernel slope
    [1e-3, 1e0], # sigma_n noise
]
lbounds = np.array([bound[0] for bound in bounds])
ubounds = np.array([bound[1] for bound in bounds])

def kernel(xp, xq, theta, include_sigma=True):
    # xp, xq are Nx1,Mx1 vectors (ln(1+xi), ln(rho_0), ln(1 + 10^3 * zeta), ln(1 + gamma))
    vec = xp - xq

    d1 = vec[1] # rho0 direction
    d2 = vec[2] # zeta direction
    d3 = vec[3] # gamma direction

    BL_kernel = theta[0] + soft_relu(-xp[1]) * soft_relu(-xq[1])
    # 0.02 was factor here before
    SE_factor = theta[1] * np.exp(-0.5 * (d1**2 / theta[2]**2 + d3**2 / theta[3]**2))
    SE_kernel = (
        SE_factor
        * soft_relu(theta[4] - soft_abs(xp[1]))
        * soft_relu(theta[4] - soft_abs(xq[1])) #* np.exp(-0.5 * d3 ** 2 / 9.0)
        * soft_relu(theta[5] - xp[3])
        * soft_relu(theta[5] - xq[3]) # correlate only low values of gamma together
    )
    gamma_kernel = theta[6] + theta[7] * xp[3] * xq[3]
    xi_kernel = theta[8] * xp[0] * xq[0]
    zeta_kernel = theta[9] * xp[2] * xq[2]
    sigma_n = theta[10]

    inner_kernel = BL_kernel * gamma_kernel**2 + SE_kernel + xi_kernel + zeta_kernel
    # return inner_kernel + sigma_n**2 * include_sigma
    return inner_kernel

def kernel_deriv(xp, xq, theta, idx, include_sigma=True):
    """compute derivative of kernel function above w.r.t. the dv index"""
    # use complex-step method here as it is very convenient
    dtheta = np.zeros((11,))
    dtheta[idx] = 1e-30
    f1 = kernel(xp, xq, theta + dtheta * 1j, include_sigma=True)
    return np.imag(f1) / 1e-30

print(f"Monte Carlo #data training {n_train} / {X.shape[0]} data points")

# randomly permute the arrays
rand_perm = np.random.permutation(N_data)
X = X[rand_perm, :]
Y = Y[rand_perm, :]

# zeta_mask = X_test[:,2] < 1.0

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
    ) + theta[10]**2 * np.eye(n_train)

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
    ) + theta[10]**2 * np.eye(n_train)

    # forward part (not gradient yet)
    alpha = np.linalg.solve(K_y, Y_train)
    term1 = (0.5 * Y_train.T @ alpha)[0,0]
    term2 = (0.5 * np.abs(np.linalg.slogdet(K_y)) )[1]
    term3 = n_train/2.0 * np.log(2.0 * np.pi)
    nmap = term1 + term2 + term3

    # now get the gradient..
    ntheta = theta.shape[0]
    grad = np.zeros((ntheta,))
    for itheta in range(ntheta):
        if can_print: print(f"\tgetting grad entry theta{itheta}")
        # get dK/dtheta_j
        if itheta < 10:
            Kgrad = np.array(
                [
                    [kernel_deriv(X_train[i, :], X_train[j, :], theta, itheta) for i in range(n_train)]
                    for j in range(n_train)
                ]
            )
        elif itheta == 10:
            Kgrad = 2 * theta[10] * np.eye(n_train)

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
    dtheta = np.random.rand(11)
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
optProb.addVarGroup("theta", 11, lower=lbounds, value=theta0, upper=ubounds)
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
