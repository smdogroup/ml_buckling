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
This time I'll try a Gaussian Process model to fit the axial critical load surrogate model
Inputs: D*, a0/b0, ln(b/h)
Output: k_x0
"""
# parse the arguments
parent_parser = argparse.ArgumentParser(add_help=False)
parent_parser.add_argument("--load", type=str)

args = parent_parser.parse_args()

assert args.load in ["Nx", "Nxy"]

load = args.load

# load the Nxcrit dataset
load_prefix = "Nx_stiffened" if load == "Nx" else "Nxy"
csv_filename = f"{load}_stiffened"
df = pd.read_csv("data/" + csv_filename + ".csv")

# extract only the model columns
# TODO : if need more inputs => could maybe try adding log(E11/E22) in as a parameter?
# or also log(E11/G12)
X = df[["log(xi)", "log(rho_0)", "log(1+10^3*zeta)", "log(1+gamma)"]].to_numpy()
Y = df["log(lam_star)"].to_numpy()
Y = np.reshape(Y, newshape=(Y.shape[0], 1))

N_data = X.shape[0]

n_train = int(0.4 * N_data)
# n_train = 1000

print(f"Monte Carlo #data training {n_train} / {X.shape[0]} data points")

# randomly permute the arrays
rand_perm = np.random.permutation(N_data)
X = X[rand_perm, :]
Y = Y[rand_perm, :]

# REMOVE THE OUTLIERS in local 4d regions
# loop over different slenderness bins
xi_bins = [[0.25 * i, 0.25 * (i + 1)] for i in range(1, 7)]
log_xi_bins = [list(np.log(np.array(xi_bin))) for xi_bin in xi_bins]
log_gamma_bins = [[0, 1], [1, 2], [2, 3], [3, 4]]

_plot = True
_plot_gamma = False
_plot_3d = True

# make a folder for the model fitting
plots_folder = os.path.join(os.getcwd(), "plots")
sub_plots_folder = os.path.join(plots_folder, csv_filename)
GP_folder = os.path.join(sub_plots_folder, "GP")
for ifolder, folder in enumerate(
    [
        plots_folder,
        sub_plots_folder,
        GP_folder,
    ]
):
    if ifolder > 0 and os.path.exists(folder):
        shutil.rmtree(folder)
    if not os.path.exists(folder):
        os.mkdir(folder)

plt.style.use(niceplots.get_style())

n_data = X.shape[0]

# split into training and test datasets
n_total = X.shape[0]
n_test = n_total - n_train
assert n_test > 100

# reorder the data
indices = [_ for _ in range(n_total)]
train_indices = np.random.choice(indices, size=n_train)
test_indices = [_ for _ in range(n_total) if not (_ in train_indices)]

X_train = X[train_indices, :]
X_test = X[test_indices, :]
Y_train = Y[train_indices, :]
Y_test = Y[test_indices, :]

# TRAIN THE MODEL HYPERPARAMETERS
# by maximizing the log marginal likelihood log p(y|X)
# or minimizing the negative log marginal likelihood
# ----------------------------------------------------

# y is the training set observations
y = Y_train

# update the local hyperparameter variables
# initial hyperparameter vector
# sigma_n, sigma_f, L1, L2, L3
theta0 = np.array([1e-1, 3e-1, -1, 0.2, 1.0, 1.0, 0.5, 2, 0.8, 1.0, 0.2, 1e-2])


def relu(x):
    return max([0.0, x])


def soft_relu(x, rho=10):
    return 1.0 / rho * np.log(1 + np.exp(rho * x))


def kernel(xp, xq, theta):
    # xp, xq are Nx1,Mx1 vectors (ln(xi), ln(rho_0), ln(1 + 10^3 * zeta), ln(1 + gamma))
    vec = xp - xq

    S1 = theta[0]
    S2 = theta[1]
    L1 = theta[2]
    S4 = theta[3]
    S5 = theta[4]
    L2 = theta[5]
    L3 = theta[6]
    S6 = theta[7]
    S7 = theta[8]
    S8 = theta[9]
    S9 = theta[10]
    # sigma_n = theta[11]

    d1 = vec[1]  # first two entries
    d2 = vec[2]
    d3 = vec[3]

    # log(xi) direction
    kernel0 = S1 ** 2 + S2 ** 2 * xp[0] * xq[0]
    # log(rho_0) direction
    kernel1 = (
        np.exp(-0.5 * (d1 ** 2 / L1 ** 2))
        * soft_relu(1 - abs(xp[1]))
        * soft_relu(1 - abs(xq[1]))
        + S4
        + S5 * soft_relu(-xp[1]) * soft_relu(-xq[1])
    )
    # log(zeta) direction
    kernel2 = S6 * np.exp(-0.5 * d2 ** 2 / L2 ** 2) + S7 * xp[2] * xq[2]
    # log(gamma) direction
    kernel3 = S8 * np.exp(-0.5 * d3 ** 2 / L3 ** 2) + S9 * xp[3] * xq[3]
    return kernel0 * kernel1 * kernel2 * kernel3


ntheta = theta0.shape[0]


def soft_abs(x, rho=1.0):
    return 1.0 / rho * np.log(np.exp(rho * x) + np.exp(-rho * x))


def soft_abs_deriv(x, rho=1.0):
    return (np.exp(rho * x) - np.exp(-rho * x)) / (np.exp(rho * x) + np.exp(-rho * x))


class MyOpt:
    def __init__(self):
        self.objective_hist = []

    def objective(self, xdict):
        """negative marginal likelihood objective function log p(y|X,theta)"""
        # computed faster using Cholesky decomposition
        # here is the training covariance matrix
        theta = xdict["theta"]

        Sigma = np.array(
            [
                [kernel(X_train[i, :], X_train[j, :], theta) for i in range(n_train)]
                for j in range(n_train)
            ]
        ) + theta[-1] ** 2 * np.eye(n_train)

        # eqn log p(y|X,theta) = - 0.5 * y^T Sigma^-1 y - 1/2 * log|Sigma| - n/2 * log(2*pi)
        L = scipy.linalg.cholesky(Sigma, lower=True)
        beta = L.T @ Y_train
        beta_vec = beta[:, 0]
        log_Sigma = 2.0 * np.sum(np.log(np.diag(L)))
        neg_obj = (
            -0.5 * np.dot(beta_vec, beta_vec)
            - 0.5 * log_Sigma
            - 0.5 * n_train * np.log(2.0 * np.pi)
        )
        obj = -1.0 * neg_obj

        # take log on the objective again
        # log(-log p(y|X,theta))
        obj2 = np.cbrt(obj)  # differentiable as obj goes from positive to negative
        # obj2 = np.log(soft_abs(obj))

        print(f"obj2 = {obj2}")
        print(f"theta = {theta}")
        funcs = {"obj": obj2}

        self.objective_hist += [obj2]

        return funcs, False

    def my_gradient(self, xdict, funcs):
        _gradient = np.zeros((ntheta,))
        theta = xdict["theta"]

        # important forward analysis states
        Sigma = np.array(
            [
                [kernel(X_train[i, :], X_train[j, :], theta) for i in range(n_train)]
                for j in range(n_train)
            ]
        ) + theta[-1] ** 2 * np.eye(n_train)

        # eqn log p(y|X,theta) = - 0.5 * y^T Sigma^-1 y - 1/2 * log|Sigma| - n/2 * log(2*pi)
        L = scipy.linalg.cholesky(Sigma, lower=True)
        temp = scipy.linalg.solve_triangular(L, Y_train, lower=True)
        alpha = scipy.linalg.solve_triangular(L.T, temp, lower=False)

        # perform complex-step to get dSigma/dtheta_j
        #  can't really complex-step full objective due to Cholesky decomposition being tricky with imaginary numbers
        for itheta in range(ntheta):
            theta_pert = theta * 1.0  # copy
            theta_pert = theta_pert.astype(np.complex128)
            theta_pert[itheta] += 1e-30 * 1j
            Sigma_final = np.array(
                [
                    [
                        kernel(X_train[i, :], X_train[j, :], theta_pert)
                        for i in range(n_train)
                    ]
                    for j in range(n_train)
                ]
            ) + theta_pert[-1] ** 2 * np.eye(n_train)

            dSigma_dth = np.imag(Sigma_final) / 1e-30

            # now compute the following formula for an analytic derivative
            # dlogp/dth = 0.5 * tr((alpha*alpha^T - Sigma^-1) * dSigma/dth) where alpha = Sigma^-1 * Y using Cholesky decomp
            term1_inside = alpha @ alpha.T @ dSigma_dth

            temp_mat = scipy.linalg.solve_triangular(L, dSigma_dth, lower=True)
            temp_mat2 = scipy.linalg.solve_triangular(L.T, temp_mat, lower=False)
            deriv = 0.5 * np.trace(term1_inside - temp_mat2)

            _gradient[itheta] = deriv

        # conver to negative objective again for maxim => minimization
        _gradient *= -1.0

        # now convert to gradient of objective 2
        obj = np.exp(funcs["obj"])
        _gradient2 = _gradient / obj  # derivative of the log(cdot) part

        print(f"_gradient2 = {_gradient2}")

        funcsSens = {
            "obj": {
                "theta": _gradient2,
            },
        }

        return funcsSens, False


my_opt = MyOpt()

# pyoptsparse problem
def run_optmization():
    # Optimization Object
    optProb = Optimization("MAP hyperparameter optimization", my_opt.objective)

    # Design Variables
    # sigma_n = theta[11]
    optProb.addVarGroup(
        "theta",
        12,
        # [S1, S2, L1, S4, S5, L2, L3, S6, S7, S8, S9, sigma_n]
        lower=np.array(
            [1e-4, 0.1, 0.1, 1e-4, 0.1, 0.1, 0.1, 1e-2, 1e-1, 1e-2, 1e-1, 1e-4]
        ),
        upper=np.array([3, 3, 5, 3, 3, 5, 5, 3, 3, 3, 3, 1e-2]),
        value=np.array([1e-1, 3e-1, 0.2, 1.0, 1.0, 0.5, 0.8, 1.0, 0.2, 0.2, 1.0, 1e-2]),
    )
    optProb.addObj("obj")
    # Optimizer
    opt = SNOPT(options={})
    optProb.printSparsity()
    return opt, optProb


if __name__ == "__main__":

    # test the forward + gradient
    # _obj,_ = objective({"theta" : theta0})
    # print(f"_obj = {_obj["obj"]:.4e}")
    # _grad,_ = my_gradient({"theta" : theta0},{})
    # for igrad,deriv in enumerate(_grad["obj"]["theta"]):
    #     print(f"_grad {igrad} = {deriv:.4e}")

    # NOTE : the objective starts out very large
    # and it seems like p(y|X,theta) is btw [0,1]
    # so log marginal likelihood should be negative and indeed log p(y|X,theta) should be small positive number for a good model
    # indeed this is true => but my initial model must not be that good since the log p(y|X,theta) is so large
    # this is correct behavior. Also try with larger sampling size of the full dataset for training & see what I get.

    # Solution
    optOptions = {}
    opt, optProb = run_optmization()
    # some bug in the analytic derivatives, just use finite differencing for now
    # have to do Cholesky decomp 13 times in the gradient and 1 in the forward normally
    # so ~12 in the forward for FD approach => results in FD being about same comp speed as analytic gradient here..
    sol = opt(optProb)
    # sol = opt(optProb, sens=my_opt.my_gradient)
    print(sol)

    # write out the data for the objective history
    import pandas as pd

    obj_hist_dict = {"obj": my_opt.objective_hist}
    df = pd.DataFrame(obj_hist_dict)
    df.to_csv("data/train_hist.csv")
